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
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

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
    log_dir = 'log'
    submission_file = 'submission.csv'

    # ##### uploading data to PoestgreDB
    # upload_to_db(yaml_file_path=yaml_file_path, data_path=data_path, table_name="adult_income")

    # ##### Read data from PoestgreDB
    train_data, test, label = read_from_db(
        yaml_file_path=yaml_file_path,
        table_name=table_name,
        label_col_name=target_col)

    test_id = test['id']

    train_data, test, label = preprocessing.run(train_data, test, label, target_col, verbose=0)

    #train, valid 스플릿
    x_train, x_valid, y_train, y_valid = train_test_split(train_data, label,
                                                        test_size=0.3,
                                                        shuffle=True,
                                                        stratify=label)



    # 모델 학습
    lgb = LGBMClassifier(n_estimators=100)
    lgb.fit(x_train, y_train)
    
    # 예측
    y_pred=lgb.predict(test)
    # 제출을 위해 test를 넣어야 하나?

    # y_pred = pd.DataFrame([test_id, y_pred], columns=['id', 'target'])
    test_id_df = pd.DataFrame(test_id, columns=['id'])
    y_pred = pd.DataFrame(y_pred, columns=['target'])
    submit = pd.concat([test_id_df, y_pred], axis=1)

    # 제출 csv 생성
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(submission_file):
        with open(submission_file, 'w'):
            pass

    # submission = pd.read_csv(data_path+'sample_submission.csv') # sample_submission.csv는 프로젝트 폴더에 두었음
    # submission['target'] = y_pred
    # submission.to_csv('submit.csv', index=False)
    # submit = pd.read_csv(data_path+'submission.csv')
    # submit['target'] = y_pred
    # submit = pd.DataFrame(y_pred, columns=['target'])
    submit.reset_index()
    submit.to_csv(os.path.dirname(os.path.realpath(__file__))+'/data/submission.csv', index=False)

    # print("정확도: {:.2f}%".format(accuracy_score(y_valid, y_pred)*100))
    # 사이트 에서의 결과와 로컬 결과가 다른건가?

    # # 전처리 결과를 다시 DB에 저장 후 불러온 후에 이어서 진행
    # upload_to_db(
    #     yaml_file_path=yaml_file_path,
    #     table_name=f"preprocessed_{table_name}",
    #     train_data=train_data,
    #     test_data=test)