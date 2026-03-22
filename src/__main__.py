import sys
from .parse import PathExtractor
from .function_scheme import *

def main():
    try:
        config = PathExtractor()
        
        print(f"Functions: {config.functions}")
        print(f"Input: {config.input}")
        print(f"Output: {config.output}")
        
        funcs: List[FunctionScheme] = SchemeLoader.load(config.functions)

        for i in funcs:
            print(i)

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
