import json
import os
from typing import Dict, List, Any
from pydantic import BaseModel
from enum import StrEnum


class ParamType(StrEnum):
    STRING = 'string'
    NUMBER = 'number'
    FLOAT = 'float'
    INTEGER = 'integer'
    BOOL = 'bool'


class FunctionParameter(BaseModel):
    """
    Represent a single parameter within a function scheme.

    Attributes:
        name (str): The name of the parameter.
        param_type (str): The data type of the parameter.
    """

    name: str
    type: ParamType


class FunctionScheme(BaseModel):
    """
    Represent the schema of a function, including its metadata and parameters.

    Attributes:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        params (Dict[str, ParamType]): A mapping of parameter names to types.
    """

    name: str
    description: str
    params: Dict[str, ParamType]

    def __init__(self, **data: Any):
        raw_params = data.pop('parameters', {})

        if not data.get('params') and raw_params:
            data['params'] = {
                name: details['type'] if isinstance(details, dict) else details
                for name, details in raw_params.items()
            }
        super().__init__(**data)

    def __repr__(self) -> str:
        params_str = ", ".join([f"{k}: {v.value}"
                                for k, v in self.params.items()])
        return f"FunctionScheme(name='{self.name}', params={{ {params_str} }})"


class SchemeLoader:
    """A utility class to load function schemes from external files."""

    @staticmethod
    def load(file_path: str) -> List[FunctionScheme]:
        """
        Load a list of FunctionScheme objects from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            List[FunctionScheme]: A list of loaded function schemes.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the JSON is invalid.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                return [FunctionScheme(**item) for item in data]
            except (json.JSONDecodeError, KeyError, Exception) as exc:
                raise ValueError(
                    f"Error loading schemes from {file_path}"
                ) from exc
