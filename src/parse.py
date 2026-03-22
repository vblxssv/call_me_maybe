import sys, os

import sys
import os
import sys



class PathExtractor:
    def __init__(self):
        self.required_flags = [
            "functions_definition",
            "input",
            "output"
        ]
        self.paths = {}
        args = sys.argv[1:]

        for i in range(len(args)):
            if args[i].startswith("--"):
                flag_name = args[i].lstrip("-")
                if flag_name in self.required_flags:
                    if i + 1 < len(args) and not args[i+1].startswith("--"):
                        self.paths[flag_name] = args[i+1]

        self._validate_presence()

    def _validate_presence(self):
        missing = [
            f"--{flag}" for flag in self.required_flags
            if flag not in self.paths
        ]
        if missing:
            raise ValueError(f"Missing required arguments: {', '.join(missing)}")

    @property
    def functions(self):
        return self.paths.get("functions_definition")

    @property
    def input(self):
        return self.paths.get("input")

    @property
    def output(self):
        return self.paths.get("output")
