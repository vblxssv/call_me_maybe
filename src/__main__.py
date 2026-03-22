import torch
from typing import List
from .function_scheme import FunctionScheme, SchemeLoader
from llm_sdk import Small_LLM_Model
from .parse import PathExtractor

def get_allowed_tokens(model, current_str: str, allowed_list: List[str]):
    """Определяет токены, чтобы достроить строку. Закрывающую кавычку НЕ добавляем здесь."""
    allowed_ids = []
    for candidate in allowed_list:
        if candidate.startswith(current_str):
            remaining = candidate[len(current_str):]
            if remaining:
                tokens = model.encode(remaining)
                # Извлекаем ID токена (учитываем разные форматы вывода SDK)
                token = tokens[0][0] if isinstance(tokens[0], (list, torch.Tensor)) else tokens[0]
                allowed_ids.append(int(token))
    return list(set(allowed_ids))

def generate_constrained_step(model, input_ids: List[int], allowed_values: List[str]):
    """Генерирует ТОЛЬКО само слово из списка, без закрывающей кавычки."""
    start_len = len(model.decode(input_ids))
    
    for _ in range(30):
        current_text = model.decode(input_ids)
        generated_part = current_text[start_len:]
        
        # Если мы уже полностью написали одно из разрешенных слов — выходим
        if any(generated_part == val for val in allowed_values):
            break
            
        logits = model.get_logits_from_input_ids(input_ids)
        if isinstance(logits, list):
            logits = torch.tensor(logits)
            
        allowed_ids = get_allowed_tokens(model, generated_part, allowed_values)
        
        if not allowed_ids:
            break
            
        mask = torch.full(logits.shape, -1e18, device=logits.device)
        mask[allowed_ids] = 0
        
        next_id = int(torch.argmax(logits + mask).item())
        input_ids.append(next_id)
    return input_ids

def generate_free_until(model, input_ids: List[int], stop_char='"', max_tokens=60):
    """Генерирует значение аргумента до кавычки."""
    for _ in range(max_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        if isinstance(logits, list):
            logits = torch.tensor(logits)
        next_id = int(torch.argmax(logits).item())
        input_ids.append(next_id)
        
        if stop_char in model.decode([next_id]):
            break
    return input_ids

def main():
    model = Small_LLM_Model()
    parse = PathExtractor()
    schemes = SchemeLoader.load(parse.functions)
    schemes_dict = {s.name: s for s in schemes}
    
    prompt = "Трахать илку"
    

    context = "Available tools:\n" + "\n".join([f"- {s.name}: {s.description}" for s in schemes])
    full_prompt = f"{context}\n\nUser: {prompt}\nJSON: "
    

    text = full_prompt + '{"name": "'
    input_ids = model.encode(text)[0].tolist()

    input_ids = generate_constrained_step(model, input_ids, list(schemes_dict.keys()))
    
    selected_name = model.decode(input_ids).split('"name": "')[-1].strip()
    
    input_ids.extend(model.encode('", "arguments": {"')[0].tolist())
    
    if selected_name in schemes_dict:
        current_scheme = schemes_dict[selected_name]
        param_names = list(current_scheme.params_dict.keys())
        
        input_ids = generate_constrained_step(model, input_ids, param_names)
        input_ids.extend(model.encode('": "')[0].tolist())
        input_ids = generate_free_until(model, input_ids, stop_char='"')
    input_ids.extend(model.encode('}}')[0].tolist())

    print("\n--- Final Result ---")
    final_output = model.decode(input_ids)
    json_part = final_output.split("JSON: ")[-1].strip()
    print(json_part)

if __name__ == "__main__":
    main()