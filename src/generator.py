import torch
from typing import List, Any, Optional
from llm_sdk import Small_LLM_Model


class JSONGenerator:
    """A generator that produces constrained JSON output using an LLM."""

    def __init__(self) -> None:
        """Initialize the generator with the model and internal state."""
        self.model = Small_LLM_Model()
        self.current_ids: List[int] = []
        self.current_text: str = ""

    def _get_encoded(self, text: str) -> List[int]:
        """Encode text into a list of token IDs.

        Args:
            text: The input string to encode.

        Returns:
            A list of integer token IDs.
        """
        raw_data: Any = self.model.encode(text)[0]
        return list(raw_data.tolist())

    def _sync_push(self, data: Any) -> None:
        """Add data to the state and synchronize text and tokens.

        Args:
            data: Either a string or a list of token IDs.
        """
        if isinstance(data, str):
            ids = self._get_encoded(data)
            self.current_ids.extend(ids)
            self.current_text += data
        else:
            self.current_ids.extend(data)
            self.current_text += self.model.decode(data)

    def _sample_constrained(self, choices: List[str]) -> Optional[int]:
        """Select the most probable token ID from a list of valid choices.

        Args:
            choices: A list of possible string continuations.

        Returns:
            The ID of the best matching token, or None if no choices exist.
        """
        logits = self.model.get_logits_from_input_ids(self.current_ids)
        best_id: Optional[int] = None
        max_val = float('-inf')

        for s in choices:
            if not s:
                continue

            encoded_ids = self.model.encode(s)[0]
            tid = int(encoded_ids[0].item())
            val = float(logits[tid])

            if val > max_val:
                max_val = val
                best_id = tid
        return best_id

    def _generate_word(self, choices: List[str]) -> str:
        """Generate tokens until the text matches one of the choices.

        Args:
            choices: A list of allowed strings.

        Returns:
            The generated string that matches a choice, or an empty string.
        """
        start_len = len(self.current_text)
        for _ in range(50):
            current = self.current_text[start_len:].replace('"', '').strip()
            if current in choices:
                return current

            candidates = [
                c[len(current):] for c in choices if c.startswith(current)
            ]
            next_id = self._sample_constrained(candidates)
            if next_id is None:
                break
            self._sync_push([next_id])
        return ""

    def _generate_until(self, stops: List[str], limit: int) -> None:
        """Greedily generate tokens until a stop sequence is encountered.

        Args:
            stops: A list of strings that trigger a generation halt.
            limit: The maximum number of tokens to generate.
        """
        for _ in range(limit):
            logits = self.model.get_logits_from_input_ids(self.current_ids)
            next_id = int(torch.as_tensor(logits).argmax().item())
            char = self.model.decode([next_id])
            if any(s in char for s in stops):
                break
            self._sync_push([next_id])

    def generate(self, prompt: str, funcs: List[Any]) -> str:
        """Generate a JSON string representing a tool call based on a prompt.

        Args:
            prompt: The user's input request.
            funcs: A list of available tool objects.

        Returns:
            A formatted JSON string or "{}" if generation fails.
        """
        self.current_ids, self.current_text = [], ""
        schemes = {f.name: f for f in funcs}

        header = "Available tools:\n"
        header += "\n".join([f"- {f.name}: {f.description}" for f in funcs])
        header += f"\n\nprompt: {prompt}\nJSON:\n"
        header += f'{{\n  "prompt": "{prompt}",\n  "name": "'

        self._sync_push(header)

        name = self._generate_word(list(schemes.keys()))
        if not name:
            return "{}"

        self._sync_push('",\n  "parameters": {')
        params = list(schemes[name].params_dict.items())

        for i, (p_name, p_type) in enumerate(params):
            self._sync_push(f'\n    "{p_name}": ')
            if "string" in p_type.lower():
                self._sync_push('"')
                self._generate_until(['"'], 50)
                self._sync_push('"')
            else:
                self._generate_until([',', ' ', '\n', '}'], 20)
            if i < len(params) - 1:
                self._sync_push(",")

        self._sync_push('\n  }\n}')

        if "JSON:\n" in self.current_text:
            return self.current_text.split("JSON:\n")[-1].strip()
        return self.current_text.strip()
