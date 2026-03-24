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
    tools_repr = "\n".join([f"- {s.name}: {s.description}" for s in funcs])
    system_prompt = f"Available tools:\n{tools_repr}\n\nprompt: {prompt}\nJSON:\n"
    input_ids = model.encode(system_prompt)[0].tolist()

    for fixed_text in ['{\n  "prompt": "', prompt, '",\n  "name": "']:
        input_ids.extend(model.encode(fixed_text)[0].tolist())

    func_names = list(schemes_dict.keys())
    selected_name = fn_add_word(model, input_ids, func_names)
    input_ids.extend(model.encode('",\n  ')[0].tolist())
    input_ids.extend(model.encode('"parameters": {')[0].tolist())
    scheme = schemes_dict[selected_name]
    param_items = list(scheme.params_dict.items())

    for i, (p_name, p_type) in enumerate(param_items):
        input_ids.extend(model.encode(f'\n    "{p_name}": ')[0].tolist())
        if "string" in p_type.lower():
            input_ids.extend(model.encode('"')[0].tolist())
            for _ in range(50):
                logits = model.get_logits_from_input_ids(input_ids)
                next_id = int(torch.argmax(torch.tensor(logits)).item())
                char = model.decode([next_id])
                if '"' in char: break
                input_ids.append(next_id)
            input_ids.extend(model.encode('"')[0].tolist())
        else:
            for _ in range(20):
                logits = model.get_logits_from_input_ids(input_ids)
                next_id = int(torch.argmax(torch.tensor(logits)).item())
                char = model.decode([next_id])
                if any(s in char for s in [',', ' ', '\n', '}']): break
                input_ids.append(next_id)
        if i < len(param_items) - 1:
            input_ids.extend(model.encode(',')[0].tolist())
    input_ids.extend(model.encode('\n  }\n}')[0].tolist())
    return model.decode(input_ids).split("JSON:")[-1].strip()


def add_json_string_to_file(json_str: str, filename: str = "data.json"):
    try:
        new_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Ошибка: строка не является валидным JSON. {e}")
        return False

    # 2. Если файла нет или он пустой — создаем новый с массивом внутри
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([new_data], f, indent=4, ensure_ascii=False)
        return True

    # 3. Если файл есть, читаем его, добавляем объект и перезаписываем
    try:
        with open(filename, 'r+', encoding='utf-8') as f:
            # Загружаем существующий массив
            data_list = json.load(f)
            
            # Убеждаемся, что это именно список
            if not isinstance(data_list, list):
                data_list = [data_list]
            
            # Добавляем новые данные
            data_list.append(new_data)
            
            # Сбрасываем каретку в начало и пишем обновленный список
            f.seek(0)
            json.dump(data_list, f, indent=4, ensure_ascii=False)
            f.truncate() # Отрезаем лишнее, если новый текст короче старого
            
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


if __name__ == "__main__":
    main()