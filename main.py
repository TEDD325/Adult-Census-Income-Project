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

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
    # upload_to_db(yaml_file_path=yaml_file_path, data_path=data_path, table_name="adult_income")

    # ##### Read data from PoestgreDB
    train_data, test, label = read_from_db(
        yaml_file_path=yaml_file_path,
        table_name=table_name,
        label_col_name=target_col)

    train_data, test, label = preprocessing.run(train_data, test, label, target_col, verbose=0)

    #train, valid 스플릿
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, label,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=label)

    # 모델 학습
    
    # lgb boost
    # lgb = LGBMClassifier()
    # lgb.fit(x_train, y_train)
    
    # xgb = XGBClassifier(tree_method='hist')
    # xgb.fit(x_train, y_train)

    from sklearn.svm import SVC
    svc = SVC()#(class_weight='balanced')
    svc.fit(x_train, y_train)

    # 예측
    # predict=lgb.predict(test) #lgb boost
    # predict=xgb.predict(test) #xg boost
    predict = svc.predict(test)
    # 제출 csv 생성
    submission = pd.read_csv('sample_submission.csv') # sample_submission.csv는 프로젝트 폴더에 두었음
    submission['target'] = predict
    submission.to_csv('submit_svc.csv', index=False)

    # 전처리 결과를 다시 DB에 저장 후 불러온 후에 이어서 진행
    # upload_to_db(
    #     yaml_file_path=yaml_file_path,
    #     table_name=f"preprocessed_{table_name}",
    #     train_data=train_data,
    #     test_data=test)