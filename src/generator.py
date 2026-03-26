import torch
from typing import List, Any, Optional
from llm_sdk import Small_LLM_Model


class JSONGenerator:
    """
    A class to generate structured JSON responses using a Language Model.

    This generator uses constrained sampling to ensure the model output
    aligns with predefined function schemes and parameter types.

    Attributes:
        model (Small_LLM_Model): The underlying language model instance.
    """

    def __init__(self) -> None:
        """Initializes the JSONGenerator with a Small_LLM_Model."""
        self.model = Small_LLM_Model()

    def _get_next_token_id(
        self,
        input_ids: List[int],
        allowed_strings: List[str]
    ) -> Optional[int]:
        """
        Predicts the next token ID constrained by a list of allowed strings.

        Args:
            input_ids (List[int]): Current sequence of token IDs.
            allowed_strings (List[str]): Strings that are valid at this step.

        Returns:
            Optional[int]: The ID of the most likely allowed token,
                or None if no tokens are valid.
        """
        logits = self.model.get_logits_from_input_ids(input_ids)
        logits_tensor = torch.tensor(logits)

        allowed_ids = [
            int(self.model.encode(s)[0][0])
            for s in allowed_strings
            if s
        ]

        if not allowed_ids:
            return None

        mask = torch.full_like(logits_tensor, -1e18)
        mask[allowed_ids] = 0
        return int(torch.argmax(logits_tensor + mask).item())

    def _add_word(
        self,
        input_ids: List[int],
        allowed_strings: List[str]
    ) -> Optional[str]:
        """
        Iteratively adds tokens to form one of the allowed strings.

        Args:
            input_ids (List[int]): Current sequence of
            token IDs (modified in-place).
            allowed_strings (List[str]): List of full strings to match.

        Returns:
            Optional[str]: The matched string if successful, else None.
        """
        start_len = len(input_ids)
        for _ in range(50):
            current_gen = self.model.decode(input_ids[start_len:])
            temp_name: str = current_gen.replace('"', '').strip()

            if temp_name in allowed_strings:
                return temp_name

            remaining = [
                f[len(temp_name):]
                for f in allowed_strings
                if f.startswith(temp_name)
            ]

            if not remaining:
                break

            next_id = self._get_next_token_id(input_ids, remaining)
            if next_id is None:
                break
            input_ids.append(next_id)
        return None

    def _generate_until(
        self,
        input_ids: List[int],
        stop_chars: List[str],
        max_tokens: int
    ) -> None:
        """
        Generates tokens until a stop character is
        encountered or limit reached.

        Args:
            input_ids (List[int]): Current sequence
            of token IDs (modified in-place).
            stop_chars (List[str]): Characters that
            trigger the end of generation.
            max_tokens (int): Maximum number of tokens to generate.
        """
        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_id = int(torch.argmax(torch.tensor(logits)).item())
            char = self.model.decode([next_id])

            if any(s in char for s in stop_chars):
                break
            input_ids.append(next_id)

    def generate(self, prompt: str, funcs: List[Any]) -> str:
        """
        Generates a JSON string representing a tool call based on the prompt.

        Args:
            prompt (str): The user input prompt.
            funcs (List[Any]): A list of FunctionScheme objects.

        Returns:
            str: A formatted JSON string with the selected
            function and arguments.
        """
        schemes_dict = {f.name: f for f in funcs}

        tools_repr = "\n".join([f"- {s.name}: {s.description}" for s in funcs])
        system_prompt = (
            f"Available tools:\n{tools_repr}\n\n"
            f"prompt: {prompt}\nJSON:\n"
        )
        input_ids = self.model.encode(system_prompt)[0].tolist()

        def add_text(text: str) -> None:
            """Helper to encode and extend input_ids."""
            input_ids.extend(self.model.encode(text)[0].tolist())

        add_text(f'{{\n  "prompt": "{prompt}",\n  "name": "')

        selected_name = self._add_word(input_ids, list(schemes_dict.keys()))
        if not selected_name:
            return "{}"

        scheme = schemes_dict[selected_name]
        add_text('",\n  "parameters": {')

        params = list(scheme.params_dict.items())
        for i, (p_name, p_type) in enumerate(params):
            add_text(f'\n    "{p_name}": ')
            if "string" in p_type.lower():
                add_text('"')
                self._generate_until(input_ids, ['"'], max_tokens=50)
                add_text('"')
            else:
                stop_list = [',', ' ', '\n', '}']
                self._generate_until(input_ids, stop_list, max_tokens=20)

            if i < len(params) - 1:
                add_text(',')

        add_text('\n  }\n}')

        full_text = self.model.decode(input_ids)
        if "JSON:\n" in full_text:
            return str(full_text.split("JSON:\n")[-1]).strip()
        return str(full_text).strip()
