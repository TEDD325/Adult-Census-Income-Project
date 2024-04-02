import yaml

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def read_yaml(file_path: str):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data