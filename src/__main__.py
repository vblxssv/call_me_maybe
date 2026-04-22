from .generator import JSONGenerator
from .path_extractor import PathExtractor
from .function_scheme import SchemeLoader, FunctionScheme
from .writer import Writer
from .prompt_reader import Reader
from typing import List
import json


def main() -> None:
    try:
        paths: PathExtractor = PathExtractor.from_sys_argv()
    except Exception as e:
        print(f"ERROR: {e}")
        return
    reader: Reader = Reader(paths.input)
    try:
        schemes: List[FunctionScheme] = SchemeLoader.load(paths.functions)
    except Exception as e:
        print(f"ERROR: {e}")
        return
    writer: Writer = Writer(paths.output)
    generator: JSONGenerator = JSONGenerator(schemes)

    for prompt in reader.stream_prompts():
        safe_prompt = json.dumps(prompt, ensure_ascii=False)
        escaped_only = safe_prompt[1:-1]
        res = generator.get_json(escaped_only)
        print(res)
        if writer.add_to_json(res):
            print("Successfully saved result.")
        else:
            print(f"Failed to save result for prompt: {prompt}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Interrupted")
