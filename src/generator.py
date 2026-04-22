from typing import List, Dict, Final

from .function_scheme import FunctionScheme, ParamType
from .lm_manager import LMManager


class JSONGenerator:
    """
    A generator that constructs JSON representations of function calls.

    This class uses an LMManager to predict which function and parameters
    should be used based on a provided natural language prompt.
    """

    def __init__(self, functions: List[FunctionScheme]) -> None:
        """
        Initialize the JSONGenerator.

        Args:
            functions: A list of available function schemes the LM can call.
        """
        self.generator: Final[LMManager] = LMManager()
        self.functions: List[FunctionScheme] = functions

    def _build_header(self, prompt: str) -> str:
        """
        Construct the initial context and JSON prefix for the model.

        Args:
            prompt: The user's input request.

        Returns:
            A formatted string containing tool descriptions and the start
            of the JSON structure.
        """
        header = "Available tools:\n"
        header += "\n".join([f"- {f.name}: {f.description}"
                             for f in self.functions])
        header += f"\n\nprompt: {prompt}\nJSON:\n"
        header += f'{{\n  "prompt": "{prompt}",\n'
        return header

    def get_json(self, prompt: str) -> str:
        """
        Generate a JSON string representing a function call via the LM.

        Args:
            prompt: The input text to process.

        Returns:
            A valid JSON-formatted string extracted from the model's output.
        """
        lm = self.generator
        lm.reset()
        schemes: Dict[str, FunctionScheme] = {f.name: f for f in self.functions
                                              }

        lm.sync_push(self._build_header(prompt))

        lm.sync_push('  "name": "')
        name: str = lm.pick_word(list(schemes.keys()))[:-1]
        chosen_func: FunctionScheme = schemes[name]

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
