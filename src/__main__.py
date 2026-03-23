import torch
from typing import List
from .function_scheme import FunctionScheme, SchemeLoader
from llm_sdk import Small_LLM_Model
from .parse import PathExtractor

def get_next_token_id(model, input_ids: List[int], allowed_strings: List[str]):
    """Маскирует логиты, разрешая только токены, ведущие к строкам из allowed_strings."""
    logits = model.get_logits_from_input_ids(input_ids)
    if isinstance(logits, list): 
        logits = torch.tensor(logits)
    
    allowed_ids = []
    for s in allowed_strings:
        if not s: continue
        tokens = model.encode(s)
        # Извлекаем ID токена (поддержка разных форматов SDK)
        if isinstance(tokens, (list, torch.Tensor)):
            t_id = tokens[0][0] if isinstance(tokens[0], (list, torch.Tensor)) else tokens[0]
        else:
            t_id = tokens[0]
        allowed_ids.append(int(t_id))
    
    allowed_ids = list(set(allowed_ids))
    if not allowed_ids: return None

    mask = torch.full(logits.shape, -1e18, device=logits.device)
    mask[allowed_ids] = 0
    return int(torch.argmax(logits + mask).item())

def main():
    model = Small_LLM_Model()
    parse = PathExtractor()
    # Загружаем схемы функций
    schemes = SchemeLoader.load(parse.functions)
    schemes_dict = {s.name: s for s in schemes}
    
    # Твой текущий промпт
    user_query = "What is the sum of 8 and 67?"
    
    # Формируем системный контекст
    tools_repr = "\n".join([f"- {s.name}: {s.description}" for s in schemes])
    system_prompt = f"Available tools:\n{tools_repr}\n\nUser: {user_query}\nJSON:\n"
    
    # 1. СТАРТ: Токенизируем начало
    input_ids = model.encode(system_prompt)[0].tolist()

    # 2. ГЕНЕРАЦИЯ СТРУКТУРЫ (Начало JSON)
    # Принудительно заставляем модель выдать правильные ключи
    for fixed_text in ['{\n  "prompt": "', user_query, '",\n  "name": "']:
        input_ids.extend(model.encode(fixed_text)[0].tolist())

    # 3. ВЫБОР ФУНКЦИИ (Constrained)
    func_names = list(schemes_dict.keys())
    temp_name = ""
    while temp_name not in func_names:
        remaining_options = [f[len(temp_name):] for f in func_names if f.startswith(temp_name)]
        next_id = get_next_token_id(model, input_ids, remaining_options)
        if next_id is None: break
        input_ids.append(next_id)
        # Декодируем только то, что после "name": "
        temp_name = model.decode(input_ids).split('"name": "')[-1].split('"')[0]

    selected_name = temp_name
    scheme = schemes_dict[selected_name]

    # 4. ПЕРЕХОД К ПАРАМЕТРАМ
    input_ids.extend(model.encode('",\n  "parameters": {')[0].tolist())

    # 5. ГЕНЕРАЦИЯ ПАРАМЕТРОВ ПО СХЕМЕ
    # Используем scheme.params_dict (ключ: тип), чтобы не зависеть от атрибутов класса
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

    # 6. ФИНАЛ: Закрываем всё
    input_ids.extend(model.encode('\n  }\n}')[0].tolist())

    # Вывод результата
    final_json = model.decode(input_ids).split("JSON:\n")[-1]
    print("\n--- Validated Constrained JSON ---")
    print(final_json)

if __name__ == "__main__":
    main()