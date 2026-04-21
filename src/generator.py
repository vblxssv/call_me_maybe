from .lm_manager import LMManager
from typing import List, Dict
from .function_scheme import FunctionScheme, ParamType


class JSONGenerator:
    def __init__(self, functions: List[FunctionScheme]):
        self.generator: LMManager = LMManager()
        self.functions: List[FunctionScheme] = functions

    def _build_header(self, prompt: str) -> str:
        header = "Available tools:\n"
        header += "\n".join([f"- {f.name}: {f.description}"
                             for f in self.functions])
        header += f"\n\nprompt: {prompt}\nJSON:\n"
        header += f'{{\n  "prompt": "{prompt}",\n'
        return header

    def get_json(self, prompt: str) -> str:
        lm = self.generator
        lm.reset()
        schemes: Dict[str, FunctionScheme] = {f.name: f for f in self.functions
                                              }

        lm.sync_push(self._build_header(prompt))

        lm.sync_push('  "name": "')
        name: str = lm.pick_word(list(schemes.keys()))[:-1]
        chosen_func: FunctionScheme = schemes[name]

        lm.sync_push(',\n  "parameters": {\n')

        total_params: int = len(list(chosen_func.params.items()))
        for i, (name, type) in enumerate(chosen_func.params.items()):
            lm.sync_push(f'    "{name}": ')
            argument: str = lm.generate_until(type)
            if ('.' not in argument and
                    (type == ParamType.NUMBER or type == ParamType.FLOAT)):
                lm.sync_push(".0")
            if i != total_params - 1:
                lm.sync_push(",\n")
        lm.sync_push('\n  }\n}')

        return lm.current_text.split('\nJSON:\n')[1]
