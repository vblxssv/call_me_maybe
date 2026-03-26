from dataclasses import dataclass
import json
import os
from typing import Generator, List, Any

@dataclass
class Reader:
    path: str

    def stream_prompts(self) -> Generator[str, None, None]:
        """
        Reads a JSON file and yields prompts one by one to save memory.
        Expected format: [{"prompt": "text"}, ...]
        """
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            print(f"Warning: File {self.path} is empty or does not exist.")
            return

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
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

    # def get_all_prompts(self) -> List[str]:
    #     """
    #     Helper method to get all prompts as a single list if needed.
    #     """
    #     return list(self.stream_prompts())