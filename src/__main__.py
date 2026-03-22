import sys
from .parse import PathExtractor
from .function_scheme import *
import sys
from typing import List
from .parse import PathExtractor
from .function_scheme import FunctionScheme, SchemeLoader


def main():
    try:
        config = PathExtractor()
        
        print(f"Functions path: {config.functions}")
        print(f"Input path: {config.input}")
        print(f"Output path: {config.output}")
        
        funcs: List[FunctionScheme] = SchemeLoader.load(config.functions)

        for func in funcs:
            print(func)

    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()