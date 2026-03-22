import torch
from typing import List
from .function_scheme import FunctionScheme, SchemeLoader, FunctionParameter
from llm_sdk import Small_LLM_Model
from .parse import PathExtractor

# --- Вспомогательные функции ---

def get_allowed_tokens(model, current_str: str, allowed_list: List[str]):
    allowed_ids = []
    for candidate in allowed_list:
        if candidate.startswith(current_str):
            remaining = candidate[len(current_str):]
            if remaining:
                tokens = model.encode(remaining)
                token = tokens[0][0] if isinstance(tokens[0], (list, torch.Tensor)) else tokens[0]
                allowed_ids.append(int(token))
    return list(set(allowed_ids))

def generate_constrained_step(model, input_ids: List[int], allowed_values: List[str]):
    start_len = len(model.decode(input_ids))
    for _ in range(30):
        current_text = model.decode(input_ids)
        generated_part = current_text[start_len:]
        if any(generated_part == val for val in allowed_values):
            break
        logits = model.get_logits_from_input_ids(input_ids)
        if isinstance(logits, list): logits = torch.tensor(logits)
        allowed_ids = get_allowed_tokens(model, generated_part, allowed_values)
        if not allowed_ids: break
        mask = torch.full(logits.shape, -1e18, device=logits.device)
        mask[allowed_ids] = 0
        next_id = int(torch.argmax(logits + mask).item())
        input_ids.append(next_id)
    return input_ids

def generate_free_until(model, input_ids: List[int], stop_chars: List[str]):
    """Генерирует значение до любого из стоп-символов."""
    for _ in range(60):
        logits = model.get_logits_from_input_ids(input_ids)
        if isinstance(logits, list): logits = torch.tensor(logits)
        next_id = int(torch.argmax(logits).item())
        
        char = model.decode([next_id])
        # Если токен содержит стоп-символ, НЕ добавляем его и выходим
        if any(s in char for s in stop_chars):
            break
            
        input_ids.append(next_id)
    return input_ids

# --- Main ---

def main():
    model = Small_LLM_Model()
    parse = PathExtractor()
    schemes = SchemeLoader.load(parse.functions)
    schemes_dict = {s.name: s for s in schemes}
    
    prompt = "calculate what is 6 + 7 = ?"
    
    tools_repr = "\n".join([f"- {s.name}: {s.description}. Params: {s.params_dict}" for s in schemes])
    full_prompt = f"Available tools:\n{tools_repr}\n\nUser: {prompt}\nJSON:\n"
    
    # 1. Начало: prompt и name
    text = full_prompt + '{\n  "prompt": "' + prompt + '",\n  "name": "'
    input_ids = model.encode(text)[0].tolist()

    # 2. ФАЗА: Имя функции (Constrained)
    input_ids = generate_constrained_step(model, input_ids, list(schemes_dict.keys()))
    selected_name = model.decode(input_ids).split('"name": "')[-1].strip()
    
    # 3. ФАЗА: Параметры (Теперь LM выбирает КЛЮЧИ сама)
    input_ids.extend(model.encode('",\n  "parameters": {')[0].tolist())
    
    if selected_name in schemes_dict:
        scheme = schemes_dict[selected_name]
        remaining_params = [p.name for p in scheme.params]
        
        # Цикл идет, пока есть незаполненные параметры
        for i in range(len(remaining_params)):
            input_ids.extend(model.encode('\n    "')[0].tolist())
            
            # --- CONSTRAINED ВЫБОР ИМЕНИ АРГУМЕНТА ---
            input_ids = generate_constrained_step(model, input_ids, remaining_params)
            
            # Определяем, какой ключ выбрала LM
            current_json = model.decode(input_ids)
            chosen_key = current_json.split('"')[-1].strip()
            
            # Убираем выбранный ключ из списка доступных, чтобы не повторяться
            if chosen_key in remaining_params:
                remaining_params.remove(chosen_key)
            
            # Дописываем синтаксис значения
            input_ids.extend(model.encode('": ')[0].tolist())
            
            # Проверяем тип
            p_type = scheme.params_dict.get(chosen_key, "string")
            is_str = "string" in p_type.lower()
            
            if is_str:
                input_ids.extend(model.encode('"')[0].tolist())
                input_ids = generate_free_until(model, input_ids, stop_chars=['"'])
                input_ids.extend(model.encode('"')[0].tolist())
            else:
                input_ids = generate_free_until(model, input_ids, stop_chars=[',', '\n', '}'])
            
            # Ставим запятую, если это не последний параметр
            if i < len(scheme.params) - 1:
                input_ids.extend(model.encode(',')[0].tolist())

    # 4. Финал
    input_ids.extend(model.encode('\n  }\n}')[0].tolist())

    print("\n--- Final Result (LM chose keys) ---")
    print(model.decode(input_ids).split("JSON:\n")[-1].strip())

if __name__ == "__main__":
    main()