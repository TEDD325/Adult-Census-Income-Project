import os
import warnings

import DataReader

warnings.filterwarnings("ignore")
import pprint

yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
target_col = 'target'

def display_info(obj: object, title: str = ""):
    pprint.pp(" ##### " + title + " ##### ")
    pprint.pp(obj)
    print(end="\n\n")

ROOT_PATH = os.getcwd()

train_data, test, label = DataReader.read_from_db(
    yaml_file_path=yaml_file_path,
    table_name="adult_income",
    label_col_name=target_col)

display_info(train_data, "Train Data loaded")

