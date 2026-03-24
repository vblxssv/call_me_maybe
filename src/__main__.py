import torch
from typing import List
from .function_scheme import FunctionScheme, SchemeLoader
from llm_sdk import Small_LLM_Model
from .parse import PathExtractor
import json, os


# получает индекс следующего токена, учитывая разрешенные слова
def fn_get_next_token_id(model, input_ids: List[int], allowed_strings: List[str]) -> int | None:
    logits = model.get_logits_from_input_ids(input_ids)
    logits = torch.tensor(logits)
    allowed_ids = [
        int(model.encode(s)[0][0])
        for s in allowed_strings
        if s
    ]
    if not allowed_ids:
        return None
    mask = torch.full_like(logits, -1e18)
    mask[allowed_ids] = 0
    return int(torch.argmax(logits + mask))

# Добавляет к промпту 1 из разрешенных слов, НЕ 1 ТОКЕН, А ИМЕННО СЛОВО
# Использует предыдущую функцию для получения токена
def fn_add_word(model, input_ids: List[int], func_names: List[str]) -> str | None:
    start_len = len(input_ids)
    for _ in range(50):
        temp_name = model.decode(input_ids[start_len:]).replace('"', '').strip()
        if temp_name in func_names:
            return temp_name
        remaining = [f[len(temp_name):] for f in func_names if f.startswith(temp_name)]
        if not remaining:
            break
        next_id = fn_get_next_token_id(model, input_ids, remaining)
        if next_id is None:
            break
        input_ids.append(next_id)
    return None


def fn_get_json(model, prompt: str, funcs: List[FunctionScheme]) -> str:
    schemes_dict = {f.name: f for f in funcs}
    
    def add_text(text: str):
        input_ids.extend(model.encode(text)[0].tolist())

    def generate_until(stop_chars: list, max_tokens: int):
        for _ in range(max_tokens):
            logits = model.get_logits_from_input_ids(input_ids)
            next_id = int(torch.argmax(torch.tensor(logits)).item())
            char = model.decode([next_id])
            if any(s in char for s in stop_chars):
                break
            input_ids.append(next_id)

    tools_repr = "\n".join([f"- {s.name}: {s.description}" for s in funcs])
    system_prompt = f"Available tools:\n{tools_repr}\n\nprompt: {prompt}\nJSON:\n"
    input_ids = model.encode(system_prompt)[0].tolist()

    add_text(f'{{\n  "prompt": "{prompt}",\n  "name": "')
    selected_name = fn_add_word(model, input_ids, list(schemes_dict.keys()))
    scheme = schemes_dict[selected_name]
    add_text('",\n  "parameters": {')

    for i, (p_name, p_type) in enumerate(scheme.params_dict.items()):
        add_text(f'\n    "{p_name}": ')        
        if "string" in p_type.lower():
            add_text('"')
            generate_until(['"'], max_tokens=50)
            add_text('"')
        else:
            generate_until([',', ' ', '\n', '}'], max_tokens=20)
        if i < len(scheme.params_dict) - 1:
            add_text(',')
    add_text('\n  }\n}')
    return model.decode(input_ids).split("JSON:\n")[-1].strip()


def add_json_string_to_file(json_str: str, filename: str = "data.json"):
    try:
        new_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Ошибка: строка не является валидным JSON. {e}")
        return False
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([new_data], f, indent=4, ensure_ascii=False)
        return True
    try:
        with open(filename, 'r+', encoding='utf-8') as f:
            data_list = json.load(f)
            if not isinstance(data_list, list):
                data_list = [data_list]
            data_list.append(new_data)
            f.seek(0)
            json.dump(data_list, f, indent=4, ensure_ascii=False)
            f.truncate()
        return True
    except Exception as e:
        print(f"Не удалось обновить файл: {e}")
        return False


def main():
    model = Small_LLM_Model()
    parse = PathExtractor()
    schemes: List[FunctionScheme] = SchemeLoader.load(parse.functions)
    prompt = "what is the sum of 6 and 7?"
    json: str = fn_get_json(model, prompt, schemes)
    add_json_string_to_file(json, parse.output)

    prompt = "how would look like word 'Masha' backwards?"
    json: str = fn_get_json(model, prompt, schemes)
    add_json_string_to_file(json, parse.output)

    prompt = "i need to concatenate 'ilka' and 'pidor'"
    json: str = fn_get_json(model, prompt, schemes)
    add_json_string_to_file(json, parse.output)


if __name__ == "__main__":
    main()
