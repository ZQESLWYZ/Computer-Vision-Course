import yaml
import os

def read_yaml(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except FileNotFoundError:
        print(f"YAML NOT FOUND, PATH:{file_path}")
        return {}
    except yaml.YAMLError:
        print(f"YAML READ ERROR")