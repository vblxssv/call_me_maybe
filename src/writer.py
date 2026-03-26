import json
import os
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Writer:
    """
    A utility class to append JSON-formatted strings to a persistent file.

    Attributes:
        path (str): The destination file path where data will be stored.
    """

    path: str

    def add_to_json(self, json_str: str) -> bool:
        """
        Parses a JSON string and appends the resulting object to a file.

        If the file already contains a JSON list, the new object is appended.
        If the file contains a single object, it is converted into a list.
        If the file doesn't exist, a new list is created.

        Args:
            json_str (str): A valid JSON-formatted string to be added.

        Returns:
            bool: True if the operation was successful, False otherwise.

        Note:
            This method rewrites the entire file to maintain valid JSON
            structure, which may be slow for very large files.
        """
        try:
            new_data: Any = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: string is not valid JSON. {e}")
            return False

        data_list: List[Any] = []

        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        data_list = [data_list]
            except Exception as e:
                print(f"Error reading file: {e}")
                return False

        data_list.append(new_data)

        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False
