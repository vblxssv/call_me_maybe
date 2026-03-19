import torch
import torch.nn.functional as F
from llm_sdk import Small_LLM_Model


def main():
    model = Small_LLM_Model()

    prompt = (
        "Ты — бешеный, прокуренный сапожник. Ты ненавидишь всех вокруг.\n"
        "Пользователь: Привет, как дела?\n"
        "Ассистент: Слышь ты, харя немытая, че приперся? У меня дел по горло, ботинки в говне, "
        "а тут ты со своими тупыми вопросами лезешь под руку, иди гуляй пока зубы целы!\n"
        "Пользователь: Эй, ты чего такой дерзкий?\n"
        "Ассистент:"
    )

    print(f"\n{prompt}", end="", flush=True)
    tokens = model.encode(prompt).flatten().tolist()

    # Ставим лимит в 100 токенов (это примерно 50-70 слов)
    for _ in range(100):
        logits_list = model.get_logits_from_input_ids(tokens)
        logits = torch.tensor(logits_list)

        # Температуру чуть выше (0.9), чтобы не зацикливалась на одном слове
        logits = logits / 0.9

        # Оставляем только топ-40 слов, чтобы не ушла в код
        v, _ = torch.topk(logits, 40)
        logits[logits < v[-1]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_token_id)
        word = model.decode([next_token_id])
        
        # Печатаем всё подряд, игнорируя точки
        print(word, end="", flush=True)

    print("\n\n[Лимит токенов исчерпан]")


if __name__ == "__main__":
    main()
