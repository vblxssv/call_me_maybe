import torch
from typing import List, Any, Optional
from llm_sdk import Small_LLM_Model


# class JSONGenerator:
#     """A generator that produces constrained JSON output using an LLM."""

#     def __init__(self) -> None:
#         """Initialize the generator with the model and internal state."""
#         self.model = Small_LLM_Model()
#         self.current_ids: List[int] = []
#         self.current_text: str = ""
#         self._token_cache: dict[str, int] = {}

#     def _get_token_id(self, s: str) -> int:
#         """Быстро достает ID первого токена из кеша или модели."""
#         if s not in self._token_cache:
#             self._token_cache[s] = int(self.model.encode(s)[0][0].item())
#         return self._token_cache[s]

#     def _get_encoded(self, text: str) -> List[int]:
#         """Encode text into a list of token IDs.

#         Args:
#             text: The input string to encode.

#         Returns:
#             A list of integer token IDs.
#         """
#         raw_data: Any = self.model.encode(text)[0]
#         return list(raw_data.tolist())

#     def _sync_push(self, data: Any) -> None:
#         """Add data to the state and synchronize text and tokens.

#         Args:
#             data: Either a string or a list of token IDs.
#         """
#         if isinstance(data, str):
#             ids = self._get_encoded(data)
#             self.current_ids.extend(ids)
#             self.current_text += data
#         else:
#             self.current_ids.extend(data)
#             self.current_text += self.model.decode(data)

#     def _sample_constrained(self, choices: List[str]) -> Optional[int]:
#         if not choices:
#             return None

#         raw_logits = self.model.get_logits_from_input_ids(self.current_ids)
#         logits_t = torch.as_tensor(raw_logits)

#         candidate_ids = [self._get_token_id(c) for c in choices if c]
#         if not candidate_ids:
#             return None

#         ids_t = torch.as_tensor(candidate_ids, device=logits_t.device)
#         best_idx = torch.argmax(logits_t[ids_t]).item()
#         return candidate_ids[best_idx]

#     def _generate_word(self, choices: List[str]) -> str:
#         """Generate tokens until the text matches one of the choices.

#         Args:
#             choices: A list of allowed strings.

#         Returns:
#             The generated string that matches a choice, or an empty string.
#         """
#         start_len = len(self.current_text)
#         for _ in range(50):
#             current = self.current_text[start_len:].replace('"', '').strip()
#             if current in choices:
#                 return current

#             candidates = [
#                 c[len(current):] for c in choices if c.startswith(current)
#             ]
#             next_id = self._sample_constrained(candidates)
#             if next_id is None:
#                 break
#             self._sync_push([next_id])
#         return ""

#     def _generate_until(self, stops: List[str], limit: int) -> None:
#         """Greedily generate tokens until a stop sequence is encountered.

#         Args:
#             stops: A list of strings that trigger a generation halt.
#             limit: The maximum number of tokens to generate.
#         """
#         for _ in range(limit):
#             logits = self.model.get_logits_from_input_ids(self.current_ids)
#             next_id = int(torch.as_tensor(logits).argmax().item())
#             char = self.model.decode([next_id])
#             if any(s in char for s in stops):
#                 break
#             self._sync_push([next_id])

#     def generate(self, prompt: str, funcs: List[Any]) -> str:
#         """Generate a JSON string representing a tool call based on a prompt.

#         Args:
#             prompt: The user's input request.
#             funcs: A list of available tool objects.

#         Returns:
#             A formatted JSON string or "{}" if generation fails.
#         """
#         self.current_ids, self.current_text = [], ""
#         schemes = {f.name: f for f in funcs}

#         header = "Available tools:\n"
#         header += "\n".join([f"- {f.name}: {f.description}" for f in funcs])
#         header += f"\n\nprompt: {prompt}\nJSON:\n"
#         header += f'{{\n  "prompt": "{prompt}",\n  "name": "'

#         self._sync_push(header)

#         name = self._generate_word(list(schemes.keys()))
#         if not name:
#             return "{}"

#         self._sync_push('",\n  "parameters": {')
#         params = list(schemes[name].params_dict.items())

#         for i, (p_name, p_type) in enumerate(params):
#             self._sync_push(f'\n    "{p_name}": ')
#             if "string" in p_type.lower():
#                 self._sync_push('"')
#                 self._generate_until(['"'], 50)
#                 self._sync_push('"')
#             else:
#                 self._generate_until([',', ' ', '\n', '}'], 20)
#             if i < len(params) - 1:
#                 self._sync_push(",")

#         self._sync_push('\n  }\n}')

#         if "JSON:\n" in self.current_text:
#             return self.current_text.split("JSON:\n")[-1].strip()
#         return self.current_text.strip()




import torch
from typing import List, Any, Optional
from llm_sdk import Small_LLM_Model


class JSONGenerator:
    """Константный генератор JSON с оптимизированным выбором токенов."""

    def __init__(self) -> None:
        self.model = Small_LLM_Model()
        self.current_ids: List[int] = []
        self.current_text: str = ""
        # Кеш для ID токенов, чтобы не вызывать encode в цикле
        self._token_cache: dict[str, int] = {}

    def _get_token_id(self, s: str) -> int:
        """Получает ID первого токена строки с использованием кеша."""
        if not s:
            return 0
        if s not in self._token_cache:
            # Кодируем строку и берем ID самого первого токена
            encoded = self.model.encode(s)[0]
            self._token_cache[s] = int(encoded[0].item())
        return self._token_cache[s]

    def _get_encoded(self, text: str) -> List[int]:
        """Кодирует текст в список ID токенов."""
        raw_data = self.model.encode(text)[0]
        return list(raw_data.tolist())

    def _sync_push(self, data: Any) -> None:
        """Добавляет данные в состояние и синхронизирует текст и токены."""
        if isinstance(data, str):
            ids = self._get_encoded(data)
            self.current_ids.extend(ids)
            self.current_text += data
        else:
            self.current_ids.extend(data)
            self.current_text += self.model.decode(data)

    def _sample_constrained(self, choices: List[str]) -> Optional[int]:
        """Выбирает наиболее вероятный токен из списка разрешенных."""
        if not choices:
            return None
            
        # ОПТИМИЗАЦИЯ 1: Если вариант один, не тратим ресурсы на модель
        if len(choices) == 1:
            return self._get_token_id(choices[0])

        # Получаем логиты один раз за шаг
        raw_logits = self.model.get_logits_from_input_ids(self.current_ids)
        logits_t = torch.as_tensor(raw_logits)

        # ОПТИМИЗАЦИЯ 2: Быстрый сбор ID токенов из кеша
        candidate_ids = [self._get_token_id(c) for c in choices if c]
        if not candidate_ids:
            return None

        # ОПТИМИЗАЦИЯ 3: Векторный поиск максимума через PyTorch
        ids_t = torch.as_tensor(candidate_ids, device=logits_t.device)
        candidate_values = logits_t[ids_t]
        best_idx = torch.argmax(candidate_values).item()
        
        return candidate_ids[int(best_idx)]

    def _generate_word(self, choices: List[str]) -> str:
        """Генерирует слово, пока оно не совпадет с одним из choices."""
        start_len = len(self.current_text)
        
        for _ in range(50):
            # Извлекаем то, что уже успели сгенерировать
            generated_part = self.current_text[start_len:].strip().replace('"', '')
            
            if generated_part in choices:
                return generated_part

            # Список возможных продолжений
            candidates = [
                c[len(generated_part):] for c in choices 
                if c.startswith(generated_part) and c != generated_part
            ]

            if not candidates:
                break
                
            # ОПТИМИЗАЦИЯ 4: Если осталось одно продолжение, просто дописываем его
            if len(candidates) == 1:
                self._sync_push(candidates[0])
                continue

            next_id = self._sample_constrained(candidates)
            if next_id is None:
                break
            self._sync_push([next_id])
            
        return self.current_text[start_len:].strip().replace('"', '')

    def _generate_until(self, stops: List[str], limit: int) -> None:
        """Жадная генерация до стоп-последовательности."""
        for _ in range(limit):
            # Оптимизированный прямой проход
            logits = torch.as_tensor(self.model.get_logits_from_input_ids(self.current_ids))
            next_id = int(logits.argmax().item())
            
            char = self.model.decode([next_id])
            if any(s in char for s in stops):
                break
            self._sync_push([next_id])

    def generate(self, prompt: str, funcs: List[Any]) -> str:
        """Основной цикл генерации JSON-структуры."""
        self.current_ids, self.current_text = [], ""
        schemes = {f.name: f for f in funcs}

        # Формируем заголовок (Header)
        header = "Available tools:\n"
        header += "\n".join([f"- {f.name}: {f.description}" for f in funcs])
        header += f"\n\nprompt: {prompt}\nJSON:\n"
        header += f'{{\n  "prompt": "{prompt}",\n  "name": "'

        self._sync_push(header)

        # Выбираем имя функции
        name = self._generate_word(list(schemes.keys()))
        if not name or name not in schemes:
            return "{}"

        self._sync_push('",\n  "parameters": {')
        
        # Генерация параметров
        params = list(schemes[name].params_dict.items())
        for i, (p_name, p_type) in enumerate(params):
            self._sync_push(f'\n    "{p_name}": ')
            
            if "string" in p_type.lower():
                self._sync_push('"')
                self._generate_until(['"'], 50)
                self._sync_push('"')
            else:
                # Для чисел/булевых значений стопы — это знаки препинания
                self._generate_until([',', ' ', '\n', '}'], 20)
            
            if i < len(params) - 1:
                self._sync_push(",")

        self._sync_push('\n  }\n}')

        # Возвращаем только JSON-часть
        if "JSON:\n" in self.current_text:
            return self.current_text.split("JSON:\n")[-1].strip()
        return self.current_text.strip()