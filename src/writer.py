"""Module for handling JSON file persistence."""

import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Writer:
    """A class to manage adding data to a JSON file.

    This class provides methods to parse JSON strings and append them
    to an existing list within a specified file.
    """

    path: str

    def add_to_json(self, json_str: str) -> bool:
        """Execute the main flow of adding data to the file.

        Parses the input string, retrieves existing data, appends the new
        entry, and saves the updated list back to the disk.

        Args:
            json_str: The JSON-formatted string to be added.

        Returns:
            True if the operation was successful, False otherwise.
        """
        new_data = self._parse_json(json_str)
        if new_data is None:
            return False

        data_list = self._read_existing_data()
        data_list.append(new_data)

        return self._write_to_file(data_list)

    def _parse_json(self, json_str: str) -> Optional[Any]:
        """Attempt to parse a string into a Python object.

        Args:
            json_str: The string to parse.

        Returns:
            The parsed object or None if a JSONDecodeError occurs.
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: string is not valid JSON. {e}")
            return None

    def _read_existing_data(self) -> List[Any]:
        """Read existing data from the file.

        If the file does not exist, is empty, or contains invalid data,
        it returns an empty list. If the file contains a single non-list
        object, it wraps it in a list to ensure compatibility.

        Returns:
            A list of objects retrieved from the file.
        """
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return []

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                content: List[Any] = json.load(f)
                return content
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading file: {e}")
            return []

    def _write_to_file(self, data: List[Any]) -> bool:
        """Write the provided list of data to the file.

        Args:
            data: The list of objects to be serialized and saved.

        Returns:
            True if writing was successful, False if an IOError occurred.
        """
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"Error writing file: {e}")
            return False
