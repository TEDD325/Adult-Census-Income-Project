import os
from os.path import join
import warnings
warnings.filterwarnings("ignore")
# from utils.Utils import *
from utils.DataReader import *
import pprint

yaml_file_path = '../info.yaml'
data_path = '../data/'
target_col = 'target'

def display_info(obj: object, title: str = ""):
    pprint.pp(" ##### " + title + " ##### ")
    pprint.pp(obj)
    print(end="\n\n")

ROOT_PATH = os.getcwd()
train_data_path = join(ROOT_PATH, '../data/', 'train.csv')

train_data, test, label = read_from_db(
    yaml_file_path=yaml_file_path,
    table_name="adult_income",
    label_col_name=target_col)

display_info(train_data, "Train Data loaded")

