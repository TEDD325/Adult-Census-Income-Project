import preprocessing
import os
import warnings
from scipy.stats import chi2_contingency
import itertools
import pprint
import DataReader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import preprocessing
from DataReader import *
from DataUploader import *

def set_initial_setting():
    warnings.filterwarnings(action='ignore')
    plt.style.use("ggplot")
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', 500)

    # seed 설정
    seed_num = 42
    np.random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)

if __name__ == "__main__":
    set_initial_setting()

    yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
    data_path = './data/'
    target_col = 'target'
    table_name = "adult_income"

    # ##### uploading data to PoestgreDB
    upload_to_db(yaml_file_path=yaml_file_path, data_path=data_path, table_name="adult_income")

    # ##### Read data from PoestgreDB
    train_data, test, label = read_from_db(
        yaml_file_path=yaml_file_path,
        table_name=table_name,
        label_col_name=target_col)

    preprocessing.run(train_data, test, label, target_col, verbose=1)

    # 전처리 결과를 다시 DB에 저장 후 불러온 후에 이어서 진행
    upload_to_db(
        yaml_file_path=yaml_file_path,
        table_name=f"preprocessed_{table_name}",
        train_data=train_data,
        test_data=test)