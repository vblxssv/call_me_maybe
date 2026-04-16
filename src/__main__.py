from .generator import JSONGenerator
from .path_extractor import PathExtractor
from .function_scheme import SchemeLoader, FunctionScheme
from .writer import Writer
from .prompt_reader import Reader
from typing import List


def main() -> None:
    try:
        parse = PathExtractor()
    except Exception as e:
        print(f"\nERROR: {e}")
        return
    print("Start lm init...")
    generator: JSONGenerator = JSONGenerator()
    print("Ended lm init...")
    try:
        schemes: List[FunctionScheme] = SchemeLoader.load(parse.functions)
    except Exception:
        print("Error: Wrong function_definitions.json format")
        return
    reader: Reader = Reader(parse.input)
    writer: Writer = Writer(parse.output)

    for prompt in reader.stream_prompts():
        print(f"Processing prompt: '{prompt[:50]}...'")
        try:
            generated_json_str = generator.generate(prompt, schemes)
            if writer.add_to_json(generated_json_str):
                print("Successfully saved result.")
            else:
                print(f"Failed to save result for prompt: {prompt}")
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            continue

    print("Processing complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ERROR: Interrupted via keyboard.")
