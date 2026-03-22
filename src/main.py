import torch
import gc
from llm_sdk import Small_LLM_Model

def main():
    model = Small_LLM_Model()
    

    text = (
        "залупа пенис хер это хиток спору нет"
    )
    token_quantity = 300 
    tokens = model.encode(text).view(-1).tolist()

    print("Генерация: ", end="", flush=True)

    try:
        for i in range(token_quantity):
            context = tokens[-512:] 
            logits = model.get_logits_from_input_ids(context)
            next_token_id = logits.index(max(logits))
            if next_token_id in [151643, 151645]:
                break
            tokens.append(next_token_id)
            word = model.decode([next_token_id])
            print(word, end="", flush=True)
            if i % 10 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nГенерация прервана пользователем.")

    print("\n\n--- Готово ---")

if __name__ == "__main__":
    main()