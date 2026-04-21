import sys
from typing import Dict, List

from pydantic import BaseModel, Field


class PathExtractor(BaseModel):
    """
    Handles extraction and validation of file paths from command line arguments

    This class serves as a container for mandatory file paths required by the
    application. It provides a factory method to parse these paths directly
    from sys.argv.

    Attributes:
        paths (Dict[str, str]): A mapping of flag names to their respective
            file system paths.
    """

    paths: Dict[str, str] = Field(description="paths")

    @classmethod
    def from_sys_argv(cls) -> "PathExtractor":
        """
        Create a PathExtractor instance by parsing command-line arguments.

        Iterates through sys.argv to find specific flags (--input, --output,
        --functions_definition) and their associated values.

        Returns:
            PathExtractor: An initialized instance containing the parsed paths.

        Raises:
            ValueError: If any of the required arguments are missing from the
                command-line input.
        """
        required_flags = ["functions_definition", "input", "output"]
        args = sys.argv[1:]
        parsed_paths: Dict[str, str] = {}

        for i, arg in enumerate(args):
            if arg.startswith("--"):
                flag_name = arg.lstrip("-")
                if flag_name in required_flags:
                    # Check if next argument exists and is not another flag
                    if i + 1 < len(args) and not args[i + 1].startswith("--"):
                        parsed_paths[flag_name] = args[i + 1]

        missing: List[str] = [
            f"--{f}" for f in required_flags if f not in parsed_paths
        ]

        if missing:
            raise ValueError(
                f"Missing required arguments: {', '.join(missing)}"
            )

        return cls(paths=parsed_paths)

    @property
    def functions(self) -> str:
        """Return the file path for functions definition."""
        return self.paths["functions_definition"]

    @property
    def input(self) -> str:
        """Return the file path for input data."""
        return self.paths["input"]

    @property
    def output(self) -> str:
        """Return the file path for output results."""
        return self.paths["output"]
