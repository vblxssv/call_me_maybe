import torch
from typing import List, Any, Optional, Dict
from llm_sdk import Small_LLM_Model


class JSONGenerator:
    """
    A constant JSON generator with optimized token selection.

    This class forces a language model to generate valid JSON by constraining
    the output vocabulary at each step based on the expected JSON schema.
    """

    def __init__(self) -> None:
        """Initialize the generator with model and caching state."""
        self.model = Small_LLM_Model()
        self.current_ids: List[int] = []
        self.current_text: str = ""
        self._token_cache: Dict[str, int] = {}

    def _get_token_id(self, s: str) -> int:
        """
        Get the ID of the first token of a string using a cache.

        Args:
            s: The string to tokenize.

        Returns:
            The ID of the first token.
        """
        if not s:
            return 0
        if s not in self._token_cache:
            encoded = self.model.encode(s)[0]
            self._token_cache[s] = int(encoded[0].item())
        return self._token_cache[s]

    def _get_encoded(self, text: str) -> List[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: Input string.

        Returns:
            List of token integers.
        """
        raw_data: Any = self.model.encode(text)[0]
        return list(raw_data.tolist())

    def _sync_push(self, data: Any) -> None:
        """
        Append data to state and synchronize text and tokens.

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
        """
        Select the most probable token from a list of allowed choices.

        Args:
            choices: A list of string candidates for the next token.

        Returns:
            The ID of the chosen token or None if no choices provided.
        """
        if not choices:
            return None
        if len(choices) == 1:
            return self._get_token_id(choices[0])

        raw_logits = self.model.get_logits_from_input_ids(self.current_ids)
        logits_t = torch.as_tensor(raw_logits)

        candidate_ids = [self._get_token_id(c) for c in choices if c]
        if not candidate_ids:
            return None

        ids_t = torch.as_tensor(candidate_ids, device=logits_t.device)
        candidate_values = logits_t[ids_t]
        best_idx = torch.argmax(candidate_values).item()

        return candidate_ids[int(best_idx)]

    def _generate_word(self, choices: List[str]) -> str:
        """
        Generate a word until it matches one of the provided choices.

        Args:
            choices: List of valid string options.

        Returns:
            The generated string.
        """
        start_len = len(self.current_text)

        for _ in range(50):
            gen_part = self.current_text[start_len:].strip().replace('"', '')

            if gen_part in choices:
                return gen_part

            candidates = [
                c[len(gen_part):] for c in choices
                if c.startswith(gen_part) and c != gen_part
            ]

            if not candidates:
                break

            if len(candidates) == 1:
                self._sync_push(candidates[0])
                continue

            next_id = self._sample_constrained(candidates)
            if next_id is None:
                break
            self._sync_push([next_id])

        return self.current_text[start_len:].strip().replace('"', '')

    def _generate_until(self, stops: List[str], limit: int) -> None:
        """
        Perform greedy generation until a stop sequence is encountered.

        Args:
            stops: List of characters/strings that trigger a stop.
            limit: Maximum number of tokens to generate.
        """
        for _ in range(limit):
            raw_logits = self.model.get_logits_from_input_ids(self.current_ids)
            logits = torch.as_tensor(raw_logits)
            next_id = int(logits.argmax().item())

            char = self.model.decode([next_id])
            if any(s in char for s in stops):
                break
            self._sync_push([next_id])

    def generate(self, prompt: str, funcs: List[Any]) -> str:
        """
        The main cycle for generating a structured JSON response.

        Args:
            prompt: The user query.
            funcs: List of available function objects with schema info.

        Returns:
            A string containing the valid JSON object.
        """
        self.current_ids, self.current_text = [], ""
        schemes = {f.name: f for f in funcs}

        header = "Available tools:\n"
        header += "\n".join([f"- {f.name}: {f.description}" for f in funcs])
        header += f"\n\nprompt: {prompt}\nJSON:\n"
        header += f'{{\n  "prompt": "{prompt}",\n  "name": "'

        self._sync_push(header)

        name = self._generate_word(list(schemes.keys()))
        if not name or name not in schemes:
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
