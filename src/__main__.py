import torch
from typing import List
from .function_scheme import FunctionScheme, SchemeLoader
from llm_sdk import Small_LLM_Model
from .parse import PathExtractor


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

# def fn_add_word(model, input_ids: List[int], func_names: List[str]) -> str | None:
#     temp_name = ""
#     while temp_name not in func_names:
#         # обрезка происходит тут через временную переменную
#         remaining_options = [f[len(temp_name):] for f in func_names if f.startswith(temp_name)]
#         next_id = fn_get_next_token_id(model, input_ids, remaining_options)
#         if next_id is None: break
#         input_ids.append(next_id)
#         temp_name = model.decode(input_ids).split('"name": "')[-1].split('"')[0]
#     return temp_name if temp_name in func_names else None

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


def main():
    model = Small_LLM_Model()
    parse = PathExtractor()
    schemes: List[FunctionScheme] = SchemeLoader.load(parse.functions)
    schemes_dict = {s.name: s for s in schemes}
    
    # Промпт
    user_query = "reverse 'pidor'"
    

    # Для контекста пишет все функции с их описанием
    tools_repr = "\n".join([f"- {s.name}: {s.description}" for s in schemes])
    system_prompt = f"Available tools:\n{tools_repr}\n\nUser: {user_query}\nJSON:\n"
    

    # Втокенизировать всю хуйню
    input_ids = model.encode(system_prompt)[0].tolist()


    # ПУНКТ 1 в блокнотике, доезжаем до выбора имени функции
    for fixed_text in ['{\n  "prompt": "', user_query, '",\n  "name": "']:
        input_ids.extend(model.encode(fixed_text)[0].tolist())



    # 2. ВЫБОР ФУНКЦИИ (Constrained)
    func_names = list(schemes_dict.keys())
    selected_name = fn_add_word(model, input_ids, func_names)
    input_ids.extend(model.encode('",\n  ')[0].tolist())


    # 3. ПЕРЕХОД К ПАРАМЕТРАМ
    input_ids.extend(model.encode('"parameters": {')[0].tolist())


    # 3. ГЕНЕРАЦИЯ ПАРАМЕТРОВ ПО СХЕМЕ
    # Используем scheme.params_dict (ключ: тип), чтобы не зависеть от атрибутов класса
    scheme = schemes_dict[selected_name]
    param_items = list(scheme.params_dict.items()) # [('a', 'number'), ('b', 'number')]
    
    for i, (p_name, p_type) in enumerate(param_items):
        # Печатаем ключ параметра
        input_ids.extend(model.encode(f'\n    "{p_name}": ')[0].tolist())
        
        # Генерируем значение
        if "string" in p_type.lower():
            input_ids.extend(model.encode('"')[0].tolist())
            # Свободная генерация до закрывающей кавычки
            for _ in range(50):
                logits = model.get_logits_from_input_ids(input_ids)
                next_id = int(torch.argmax(torch.tensor(logits)).item())
                char = model.decode([next_id])
                if '"' in char: break
                input_ids.append(next_id)
            input_ids.extend(model.encode('"')[0].tolist())
        else:
            # Для чисел — генерируем до запятой или конца блока
            for _ in range(20):
                logits = model.get_logits_from_input_ids(input_ids)
                next_id = int(torch.argmax(torch.tensor(logits)).item())
                char = model.decode([next_id])
                if any(s in char for s in [',', ' ', '\n', '}']): break
                input_ids.append(next_id)

        # Ставим запятую, если параметр не последний
        if i < len(param_items) - 1:
            input_ids.extend(model.encode(',')[0].tolist())

    # 4. ФИНАЛ: Закрываем всё
    input_ids.extend(model.encode('\n  }\n}')[0].tolist())

    # Вывод результата
    # final_json = model.decode(input_ids).split("JSON:\n")[-1]
    print("\n--- Validated Constrained JSON ---")
    print(model.decode(input_ids))

if __name__ == "__main__":
    main()