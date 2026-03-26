from dataclasses import dataclass
import json
import os

@dataclass
class Writer:
    path: str

    def add_to_json(self, json_str: str) -> bool:
        try:
            new_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error: string is not valid JSON. {e}")
            return False

        data_list = []

        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        data_list = [data_list]
            except Exception as e:
                print(f"Error reading file: {e}")
                return False

        data_list.append(new_data)
        
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing file: {e}")
            return False
