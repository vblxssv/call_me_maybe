import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class FunctionParameter:
    """
    Represents a single parameter within a function scheme.

    Attributes:
        name (str): The name of the parameter.
        param_type (str): The data type of the parameter.
    """

    name: str
    param_type: str

    def __repr__(self) -> str:
        """
        Returns a string representation of the parameter.

        Returns:
            str: Formatted string as 'name: type'.
        """
        return f"{self.name}: {self.param_type}"


class FunctionScheme:
    """
    Represents the schema of a function, including its metadata and parameters.

    Args:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        parameters (dict): A dictionary containing parameter definitions.

    Attributes:
        name (str): The name of the function.
        description (str): The function's description.
        params (List[FunctionParameter]): A list of FunctionParameter objects.
        params_dict (Dict[str, str]): A mapping of parameter names to types.
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]
                 ):
        self.name = name
        self.description = description
        self.params: List[FunctionParameter] = [
            FunctionParameter(p_name, p_info['type'])
            for p_name, p_info in parameters.items()
        ]
        self.params_dict: Dict[str, str] = {
            p.name: p.param_type for p in self.params
        }

    def get_type(self, param_name: str) -> str:
        """
        Retrieves the type of a specific parameter.

        Args:
            param_name (str): The name of the parameter to look up.

        Returns:
            str: The type of the parameter, or "string" if not found.
        """
        return self.params_dict.get(param_name, "string")

    def __repr__(self) -> str:
        """
        Returns a string representation of the FunctionScheme.

        Returns:
            str: Detailed string including function name and parameters.
        """
        params_str = ", ".join([repr(p) for p in self.params])
        return f"FunctionScheme(name='{self.name}', params=[{params_str}])"


class SchemeLoader:
    """
    A utility class to load function schemes from external files.
    """

    @staticmethod
    def load(file_path: str) -> List[FunctionScheme]:
        """
        Loads a list of FunctionScheme objects from a JSON file.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            List[FunctionScheme]: A list of loaded function schemes.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the JSON is invalid or the root is not a list.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data: list[dict[str, Any]] = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON format in: {file_path}"
                ) from exc
        return [
            FunctionScheme(
                name=item['name'],
                description=item['description'],
                parameters=item['parameters']
            )
            for item in data
        ]
