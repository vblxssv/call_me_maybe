from llm_sdk import Small_LLM_Model
from typing import List


def main():
    model = Small_LLM_Model()
    text = """<|im_start|>system
                Ты — полезный помощник. Отвечай кратко и по делу.<|im_end|>
                <|im_start|>user
                Какое яблоко объективно считается большим? Назови конкретные критерии (вес, диаметр).<|im_end|>
                <|im_start|>assistant"""
                    
    print(f"Генерирую продолжение для: {text}...")

    for _ in range(200):  # Попробуем сгенерировать 10 токенов
        # 1. Кодируем текущий текст
        tokens_tensor = model.encode(text)
        tokens_list = tokens_tensor[0].tolist()

        # 2. Получаем логиты
        logits = model.get_logits_from_input_ids(tokens_list)

        # 3. Берем самый вероятный токен
        next_token_id = logits.index(max(logits))

        # 4. Превращаем его в текст
        next_token_text = model._tokenizer.decode([next_token_id])

        # 5. Добавляем к общей фразе и идем на новый круг
        text += next_token_text
        
        # Печатаем по кусочкам, чтобы видеть процесс
        print(f"Добавил токен: '{next_token_text}' (ID: {next_token_id})")

    print("=" * 40)
    print(f"\nИтоговый результат: {text}")
    print("=" * 40)



if __name__ == "__main__":
    main()
