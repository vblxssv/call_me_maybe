import sys
from typing import List
from .parse import PathExtractor
from .function_scheme import FunctionScheme, SchemeLoader
from llm_sdk import Small_LLM_Model

def main():
    config = PathExtractor()
    funcs: List[FunctionScheme] = SchemeLoader.load(config.functions)
    
    model = Small_LLM_Model()
    
    text = '{"prompt": "What is the sum of 228 and 1488?", "name": "fn_add_numbers", "parameters": {"a": 228.0, "b": '
    
    tokens = 10

    input_ids = model.encode(text)[0].tolist()

    for _ in range(tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        input_ids.append(logits.index(max(logits)))

    print(model.decode(input_ids))


if __name__ == "__main__":
    main()
