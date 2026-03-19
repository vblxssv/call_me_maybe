






# Используем uv для скорости и надежности
UV = uv
PYTHON = $(UV) run python

# Установка и синхронизация зависимостей
install:
	$(UV) sync

# Запуск основного скрипта через окружение uv
run:
	$(PYTHON) main.py

# Отладка (pdb) внутри окружения uv
debug:
	$(PYTHON) -m pdb main.py

# Очистка мусора (кэши python и mypy)
clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .venv

# Стандартная проверка кода
lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

# Строгая проверка кода
lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
