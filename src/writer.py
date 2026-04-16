"""Module for handling JSON file persistence."""

import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional
from pydantic import BaseModel, ValidationError, TypeAdapter


class DataEntry(BaseModel):
    """
    Schema for a single data entry.
    Adjust the fields here to match your actual JSON structure.
    If you want to allow any dict, use: dict[str, Any]
    """
    prompt: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None


@dataclass
class Writer:
    """A class to manage adding data to a JSON file."""

    path: str

    def add_to_json(self, json_str: str) -> bool:
        """Execute the main flow of adding data to the file."""
        new_data = self._parse_json(json_str)
        if new_data is None:
            return False

        data_list = self._read_existing_data()
        data_list.append(new_data)

        return self._write_to_file(data_list)

    def _parse_json(self, json_str: str) -> Optional[dict[str, Any]]:
        """Parse string using Pydantic model validation."""
        try:
            return DataEntry.model_validate_json(json_str).model_dump()
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Error: string is not valid JSON or schema mismatch. {e}")
            return None

    def _read_existing_data(self) -> List[dict[str, Any]]:
        """Read and validate existing list from file."""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return []

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                adapter = TypeAdapter(List[DataEntry])
                validated_list = adapter.validate_python(raw_data)
                return [item.model_dump() for item in validated_list]
        except (ValidationError, json.JSONDecodeError, IOError) as e:
            print(f"Error reading file or data is invalid: {e}")
            return []

    def _write_to_file(self, data: List[dict[str, Any]]) -> bool:
        """Write the provided list to the file."""
        try:
            adapter = TypeAdapter(List[DataEntry])
            valid_data = adapter.validate_python(data)
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    [item.model_dump() for item in valid_data],
                    f,
                    indent=4,
                    ensure_ascii=False
                )
            return True
        except (ValidationError, IOError) as e:
            print(f"Error writing file: {e}")
            return False
