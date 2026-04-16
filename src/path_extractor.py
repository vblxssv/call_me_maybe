import sys
from typing import Dict, Any
from pydantic import BaseModel, Field, model_validator


class PathExtractor(BaseModel):
    """
    Parses and validates command-line arguments for file paths using Pydantic.

    Attributes:
        paths (Dict[str, str]): A mapping of flag names to their file paths.
    """

    paths: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def extract_from_sys_argv(cls, data: Any) -> Any:
        """
        Extract paths from sys.argv if not explicitly provided.

        Args:
            data (Any): Initial data for the model.

        Returns:
            Any: Data dictionary with populated paths.
        """
        if isinstance(data, dict) and not data.get("paths"):
            required_flags = ["functions_definition", "input", "output"]
            args = sys.argv[1:]
            parsed_paths = {}

            for i in range(len(args)):
                if args[i].startswith("--"):
                    flag_name = args[i].lstrip("-")
                    if flag_name in required_flags:
                        if (i + 1 < len(args)
                                and not args[i + 1].startswith("--")):
                            parsed_paths[flag_name] = args[i + 1]
            data["paths"] = parsed_paths
        return data

    @model_validator(mode="after")
    def validate_required_paths(self) -> "PathExtractor":
        """
        Ensure all mandatory paths are present in the dictionary.

        Returns:
            PathExtractor: The validated instance.

        Raises:
            ValueError: If any required flag is missing.
        """
        required_flags = ["functions_definition", "input", "output"]
        missing = [f"--{flag}"
                   for flag in required_flags if flag not in self.paths]

        if missing:
            raise ValueError(f"Missing required "
                             f"arguments: {', '.join(missing)}")
        return self

    @property
    def functions(self) -> str:
        """Return the path to the functions definition file."""
        return self.paths["functions_definition"]

    @property
    def input(self) -> str:
        """Return the path to the input file."""
        return self.paths["input"]

    @property
    def output(self) -> str:
        """Return the path to the output file."""
        return self.paths["output"]
