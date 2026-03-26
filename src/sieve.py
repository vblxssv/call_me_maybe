from enum import Enum, auto

class State(Enum):
    START_OBJECT = auto()      # Ожидаем '{'
    KEY_PROMPT = auto()        # Пишем "prompt": "
    VALUE_PROMPT = auto()      # Генерируем сам текст промпта
    KEY_NAME = auto()          # Пишем ", "name": "
    VALUE_NAME = auto()        # Выбираем имя функции из Registry
    KEY_PARAMS = auto()        # Пишем ", "parameters": {
    VALUE_PARAMS_KEY = auto()  # Пишем имя аргумента (н-р, "a": )
    VALUE_PARAMS_VAL = auto()  # Генерируем значение (цифры или строку)
    END_PARAMS = auto()        # Пишем '}'
    END_OBJECT = auto()        # Пишем '}'






