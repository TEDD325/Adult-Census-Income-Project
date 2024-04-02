import yaml

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def read_yaml(file_path: str):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    @staticmethod
    def check_lib_version():
        print("numpy version: {}".format(np.__version__))
        print("pandas version: {}".format(pd.__version__))
        print("matplotlib version: {}".format(matplotlib.__version__))
        print("scikit-learn version: {}".format(sklearn.__version__))
        print("xgboost version: {}".format(xgb.__version__))
        print("lightgbm version: {}".format(lgb.__version__))
        print("catboost version: {}".format(catboost.__version__))

    @staticmethod
    def display_info(obj: object, title: str = ""):
        pprint.pp(" ##### " + title + " ##### ")
        pprint.pp(obj)
        print(end="\n\n")