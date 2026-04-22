from typing import List, Dict, Any
from pydantic import BaseModel, PrivateAttr

from .function_scheme import FunctionScheme, ParamType
from .lm_manager import LMManager


class JSONGenerator(BaseModel):
    """
    A generator that uses a Language Model to produce structured JSON outputs
    based on a set of available function schemes.
    """
    functions: List[FunctionScheme]
    _generator: LMManager = PrivateAttr(default_factory=LMManager)
    _schemes: Dict[str, FunctionScheme]

    def __init__(self, functions: List[FunctionScheme], **data: Any) -> None:
        """
        Initializes the JSONGenerator with a list of function schemes and
        maps them by name for quick lookup.
        """
        super().__init__(functions=functions, **data)
        self._schemes = {f.name: f for f in self.functions}

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic post-initialization hook to ensure the internal scheme
        mapping is populated.
        """
        self._schemes = {f.name: f for f in self.functions}

    def _build_header(self, prompt: str) -> str:
        """
        Constructs the initial system prompt and the start of the JSON
        structure to guide the language model.
        """
        header = "Available tools:\n"
        header += "\n".join([f"- {f.name}: {f.description}"
                             for f in self.functions])
        header += f"\n\nprompt: {prompt}\nJSON:\n"
        header += f'{{\n  "prompt": "{prompt}",\n'
        return header

    def get_json(self, prompt: str) -> str:
        """
        Generates a valid JSON string by forcing the language model to pick
        a valid function name and generate parameters according to the
        defined schema.

        Args:
            prompt: The user input describing the task to be performed.

        Returns:
            A string containing the formatted JSON response.
        """
        lm = self._generator
        lm.reset()

        lm.sync_push(self._build_header(prompt))

        lm.sync_push('  "name": "')
        name: str = lm.pick_word(list(self._schemes.keys()))[:-1]
        chosen_func: FunctionScheme = self._schemes[name]

        lm.sync_push(',\n  "parameters": {\n')

        params_list = list(chosen_func.params.items())
        total_params: int = len(params_list)

        for i, (p_name, p_type) in enumerate(params_list):
            lm.sync_push(f'    "{p_name}": ')
            argument: str = lm.generate_until(p_type)

            is_numeric = p_type in (ParamType.NUMBER, ParamType.FLOAT)
            if is_numeric and "." not in argument:
                lm.sync_push(".0")

            if i != total_params - 1:
                lm.sync_push(",\n")

        lm.sync_push('\n  }\n}')

        return lm.current_text.split('\nJSON:\n')[1]
