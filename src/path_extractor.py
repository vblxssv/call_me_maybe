import sys
from typing import Dict, List


class PathExtractor:
    """
    Parses and validates command-line arguments for file paths.

    This class extracts specific flags defined in `required_flags` from
    the system arguments and ensures all mandatory paths are provided.

    Attributes:
        required_flags (List[str]): List of flag names that must be present.
        paths (Dict[str, str]): A mapping of flag names to their file paths.
    """

    def __init__(self) -> None:
        """
        Initializes the PathExtractor and parses sys.argv.

        Raises:
            ValueError: If any of the required flags are missing from the
                command-line arguments.
        """
        self.required_flags: List[str] = [
            "functions_definition",
            "input",
            "output"
        ]
        self.paths: Dict[str, str] = {}
        args = sys.argv[1:]

        for i in range(len(args)):
            if args[i].startswith("--"):
                flag_name = args[i].lstrip("-")
                if flag_name in self.required_flags:
                    # Check if next argument exists and is not another flag
                    if i + 1 < len(args) and not args[i + 1].startswith("--"):
                        self.paths[flag_name] = args[i + 1]

        self._validate_presence()

    def _validate_presence(self) -> None:
        """
        Checks if all required flags have been successfully parsed.

        Raises:
            ValueError: If one or more required flags are missing.
        """
        missing = [
            f"--{flag}" for flag in self.required_flags
            if flag not in self.paths
        ]
        if missing:
            raise ValueError(
                f"Missing required arguments: {', '.join(missing)}"
            )

    @property
    def functions(self) -> str:
        """
        Returns the path to the functions definition file.

        Returns:
            str: File path string.

        Raises:
            KeyError: If the path was not found (should be caught by init).
        """
        return self.paths["functions_definition"]

    @property
    def input(self) -> str:
        """
        Returns the path to the input file.

        Returns:
            str: File path string.

        Raises:
            KeyError: If the path was not found (should be caught by init).
        """
        return self.paths["input"]

    @property
    def output(self) -> str:
        """
        Returns the path to the output file.

        Returns:
            str: File path string.

        Raises:
            KeyError: If the path was not found (should be caught by init).
        """
        return self.paths["output"]
