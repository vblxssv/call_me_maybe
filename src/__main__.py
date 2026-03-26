from . import *
from typing import List


def main():
    generator: JSONGenerator = JSONGenerator()
    parse = PathExtractor()
    schemes: List[FunctionScheme] = SchemeLoader.load(parse.functions)
    writer: Writer = Writer(parse.output)
    reader: Reader = Reader(parse.input)

    for prompt in reader.stream_prompts():
        print(f"Processing prompt: '{prompt[:50]}...'")
        try:
            generated_json_str = generator.generate(prompt, schemes)
            if writer.add_to_json(generated_json_str):
                print(f"Successfully saved result.")
            else:
                print(f"Failed to save result for prompt: {prompt}")
        except Exception as e:
            print(f"An error occurred during generation: {e}")
            continue

    print("Processing complete.")


if __name__ == "__main__":
    main()
