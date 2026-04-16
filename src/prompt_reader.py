import json
import os
from dataclasses import dataclass
from typing import Dict, Generator, Any, List
from pydantic import BaseModel, ValidationError


class PromptItem(BaseModel):
    """
    Internal schema for prompt validation.

    Attributes:
        prompt (str): The prompt text.
    """

    prompt: str


@dataclass
class Reader:
    """
    A utility class to read and stream prompts from a JSON file.

    Attributes:
        path (str): The file path to the JSON source.
    """

    path: str

    def stream_prompts(self) -> Generator[str, None, None]:
        """
        Read a JSON file and yield prompts one by one.

        Expected format: [{"prompt": "text"}, ...]

        Yields:
            str: The 'prompt' value from each valid item.
        """
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            print(f"Warning: File {self.path} is empty or does not exist.")
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data: List[Dict[str, Any]] = json.load(f)

            for item in data:
                try:
                    # Validating using Pydantic model internally
                    valid_item = PromptItem.model_validate(item)
                    yield valid_item.prompt
                except (ValidationError, TypeError):
                    print(f"Skipping invalid item: {item}")

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from {self.path}. {e}")
        except Exception as e:
            print(f"Unexpected error while reading {self.path}: {e}")
