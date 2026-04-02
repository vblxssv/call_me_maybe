import torch
from typing import List, Any, Optional
from llm_sdk import Small_LLM_Model


class JSONGenerator:
    """
    A modular JSON generator for constrained LLM sampling.
    Designed to be extensible and compliant with clean code principles.
    """

    def __init__(self) -> None:
        self.model = Small_LLM_Model()

    def _get_encoded(self, text: str) -> List[int]:
        raw_data: Any = self.model.encode(text)[0]
        return list(raw_data.tolist())

    def _extend_ids(self, ids: List[int], text: str) -> None:
        ids.extend(self._get_encoded(text))

    def _sample_constrained(self, ids: List[int],
                            choices: List[str]) -> Optional[int]:
        """Calculates logits and applies a mask for allowed strings."""
        logits = torch.tensor(self.model.get_logits_from_input_ids(ids))
        allowed_ids = [int(self.model.encode(s)[0][0]) for s in choices if s]

        if not allowed_ids:
            return None

        mask = torch.full_like(logits, -1e18)
        mask[allowed_ids] = 0
        return int(torch.argmax(logits + mask).item())

    def _generate_word(self, ids: List[int], choices: List[str]) -> str:
        """Generates a token sequence that must match one of the choices."""
        start_idx = len(ids)
        for _ in range(50):
            current: str = self.model.decode(
                ids[start_idx:]).replace('"', '').strip()
            if current in choices:
                return current

            candidates = [
                c[len(current):] for c in choices if c.startswith(current)
                ]
            next_id = self._sample_constrained(ids, candidates)
            if next_id is None:
                break
            ids.append(next_id)
        return ""

    def _generate_value(self, ids: List[int], p_type: str) -> None:
        """Generates a field value based on its expected type."""
        if "string" in p_type.lower():
            self._extend_ids(ids, '"')
            self._generate_until(ids, ['"'], 50)
            self._extend_ids(ids, '"')
        else:
            self._generate_until(ids, [',', ' ', '\n', '}'], 20)

    def _generate_until(self, ids: List[int],
                        stops: List[str], limit: int) -> None:
        """Greedy generation until a stop character is encountered."""
        for _ in range(limit):
            logits = torch.tensor(self.model.get_logits_from_input_ids(ids))
            next_id = int(torch.argmax(logits).item())
            if any(s in self.model.decode([next_id]) for s in stops):
                break
            ids.append(next_id)

    def generate(self, prompt: str, funcs: List[Any]) -> str:
        """Main entry point: orchestrates the JSON structure generation."""
        schemes = {f.name: f for f in funcs}
        input_ids = self._get_encoded(self._build_prompt(funcs, prompt))

        self._extend_ids(input_ids, f'{{\n  "prompt": "{prompt}",\n  "name": "'
                         )

        name = self._generate_word(input_ids, list(schemes.keys()))
        if not name:
            return "{}"

        self._extend_ids(input_ids, '",\n  "parameters": {')
        params = list(schemes[name].params_dict.items())

        for i, (p_name, p_type) in enumerate(params):
            self._extend_ids(input_ids, f'\n    "{p_name}": ')
            self._generate_value(input_ids, p_type)
            if i < len(params) - 1:
                self._extend_ids(input_ids, ",")

        self._extend_ids(input_ids, '\n  }\n}')
        return self._format_output(input_ids)

    def _build_prompt(self, funcs: List[Any], prompt: str) -> str:
        tools = "\n".join([f"- {f.name}: {f.description}" for f in funcs])
        return f"Available tools:\n{tools}\n\nprompt: {prompt}\nJSON:\n"

    def _format_output(self, ids: List[int]) -> str:
        text = self.model.decode(ids)
        res: str = ''
        if "JSON:\n" in text:
            res = text.split("JSON:\n")[-1].strip()
        else:
            res = text.strip()
        return res
