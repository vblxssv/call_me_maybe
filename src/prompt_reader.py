import json
import os
from dataclasses import dataclass
from typing import Any, Generator


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
        Reads a JSON file and yields prompts one by one to save memory.

        Expected format: [{"prompt": "text"}, ...]

        Yields:
            str: The 'prompt' value from each valid dictionary in a JSON list.

        Note:
            Skips invalid items and logs warnings if the file is missing,
            empty, or has an incorrect structure.
        """
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            print(f"Warning: File {self.path} is empty or does not exist.")
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data: Any = json.load(f)

            if not isinstance(data, list):
                print(f"Error: JSON root in {self.path} must be a list.")
                return

            for item in data:
                if isinstance(item, dict) and "prompt" in item:
                    yield item["prompt"]
                else:
                    print(f"Skipping invalid item: {item}")

        except json.JSONDecodeError as e:
            print(f"Error: Failed to decode JSON from {self.path}. {e}")
        except Exception as e:
            print(f"Unexpected error while reading {self.path}: {e}")
