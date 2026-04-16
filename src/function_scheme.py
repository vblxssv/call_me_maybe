import json
import os
from typing import Dict, List, Any
from pydantic import BaseModel, Field


class FunctionParameter(BaseModel):
    """
    Represent a single parameter within a function scheme.

    Attributes:
        name (str): The name of the parameter.
        param_type (str): The data type of the parameter.
    """

    name: str
    param_type: str = Field(alias="type")

    def __repr__(self) -> str:
        """Return a formatted string representation as 'name: type'."""
        return f"{self.name}: {self.param_type}"

    model_config = {"populate_by_name": True}


class FunctionScheme(BaseModel):
    """
    Represent the schema of a function, including its metadata and parameters.

    Attributes:
        name (str): The name of the function.
        description (str): A brief description of what the function does.
        params (List[FunctionParameter]): A list of FunctionParameter objects.
        params_dict (Dict[str, str]): A mapping of parameter names to types.
    """

    name: str
    description: str
    params: List[FunctionParameter] = Field(default_factory=list)
    params_dict: Dict[str, str] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        """
        Initialize the scheme and reconstruct params and params_dict.

        This ensures logic parity with the original non-Pydantic version.
        """
        if "parameters" in data:
            raw_params = data["parameters"]
            data["params"] = [
                FunctionParameter(name=p_name, type=p_info['type'])
                for p_name, p_info in raw_params.items()
            ]
            data["params_dict"] = {
                p.name: p.param_type for p in data["params"]
            }
        super().__init__(**data)

    def get_type(self, param_name: str) -> str:
        """
        Retrieve the type of a specific parameter.

        Args:
            param_name (str): The name of the parameter to look up.

        Returns:
            str: The type of the parameter, or "string" if not found.
        """
        return self.params_dict.get(param_name, "string")

    def __repr__(self) -> str:
        """Return a detailed string representation of the FunctionScheme."""
        params_str = ", ".join([repr(p) for p in self.params])
        return f"FunctionScheme(name='{self.name}', params=[{params_str}])"


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
