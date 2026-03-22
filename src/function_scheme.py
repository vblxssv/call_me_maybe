from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class FunctionParameter:
    name: str
    param_type: str

    def __repr__(self):
        return f"{self.name}: {self.param_type}"

class FunctionScheme:
    def __init__(self, name: str, description: str, parameters: dict):
        self.name = name
        self.description = description
        self.params: List[FunctionParameter] = [
            FunctionParameter(p_name, p_info['type']) 
            for p_name, p_info in parameters.items()
        ]
        self.params_dict: Dict[str, str] = {p.name: p.param_type for p in self.params}

    def get_type(self, param_name: str) -> str:
        return self.params_dict.get(param_name, "string")

    def __repr__(self):
        params_str = ", ".join([repr(p) for p in self.params])
        return f"FunctionScheme(name='{self.name}', params=[{params_str}])"


class SchemeLoader:
    @staticmethod
    def load(file_path: str) -> List[FunctionScheme]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON root must be a list")

        return [
            FunctionScheme(
                name=item['name'],
                description=item['description'],
                parameters=item['parameters']
            )
            for item in data
        ]
