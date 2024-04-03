# import preprocessing
# import os
# import warnings
# from scipy.stats import chi2_contingency
# import itertools
# import pprint
# import DataReader
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
# import preprocessing
# from DataReader import *
# from DataUploader import *
# from sklearn.metrics import accuracy_score
#
# from sklearn.model_selection import train_test_split
#
# from lightgbm import LGBMClassifier
# from sklearn.svm import SVC
# from catboost import CatBoostClassifier
#
#
# def set_initial_setting():
#     warnings.filterwarnings(action='ignore')
#     plt.style.use("ggplot")
#     pd.set_option('display.max_rows', 30)
#     pd.set_option('display.max_columns', 500)
#
#     # seed 설정
#     seed_num = 42
#     np.random.seed(seed_num)
#     os.environ['PYTHONHASHSEED'] = str(seed_num)
#
# if __name__ == "__main__":
#     set_initial_setting()
#
#     yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
#     data_path = './data/'
#     target_col = 'target'
#     table_name = "adult_income"
#     log_dir = 'log'
#     submission_file = 'submission.csv'
#
#     # ##### uploading data to PoestgreDB
#     # upload_to_db(yaml_file_path=yaml_file_path, data_path=data_path, table_name="adult_income")
#
#     # ##### Read data from PoestgreDB
#     train_data, test, label = read_from_db(
#         yaml_file_path=yaml_file_path,
#         table_name=table_name,
#         label_col_name=target_col)
#
#     test_id = test['id']
#
#     x_train = train_data.reset_index(drop=True)
#     label = label.reset_index(drop=True)
#
#     train_data, test, label = preprocessing.run(train_data, test, label, target_col, verbose=0)
#
#     #train, valid 스플릿
#     x_train, x_valid, y_train, y_valid = train_test_split(train_data, label,
#                                                         test_size=0.3,
#                                                         shuffle=True,
#                                                         stratify=label)
#
#
#
#     # 모델 학습
#     # lgb = LGBMClassifier(n_estimators=100)
#     # lgb.fit(x_train, y_train)
#     # y_pred = lgb.predict(test)
#     # svc = SVC(class_weight='balanced')
#     # svc.fit(x_train, y_train)
#     # y_pred = svc.predict(x_valid)
#     catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.001, depth=10, random_state=42)
#     catboost_model.fit(x_train, y_train, verbose=100)
#     y_pred = catboost_model.predict(test)
#     # accuracy = accuracy_score(y_valid, y_pred)
#
#     # y_pred = pd.DataFrame([test_id, y_pred], columns=['id', 'target'])
#     test_id_df = pd.DataFrame(test_id, columns=['id'])
#     y_pred = pd.DataFrame(y_pred, columns=['target'])
#     submit = pd.concat([test_id_df, y_pred], axis=1)
#     submit = submit.dropna()
#     submit = submit.drop_duplicates(subset=['id'])
#
#
#     submission = pd.read_csv(os.path.dirname(os.path.realpath(__file__))+'/data/sample_submission.csv') # sample_submission.csv는 프로젝트 폴더에 두었음
#     # submission = submission.drop('target', axis=1)
#     # submit_with_target = pd.merge(submission, submit, on='id', how='inner')
#     # merged_df = submit_with_target.drop_duplicates(subset=['id'])
#     merged_df = submission['target'] = submit
#     merged_df.to_csv(os.path.dirname(os.path.realpath(__file__))+'/data/submission.csv', index=False)
#     # submit = pd.read_csv(data_path+'submission.csv')
#     # submit['target'] = y_pred
#     # submit = pd.DataFrame(y_pred, columns=['target'])
#     # submit.reset_index()
#     # submit.to_csv(os.path.dirname(os.path.realpath(__file__))+'/data/submission.csv', index=False)
#
#     # print("정확도: {:.2f}%".format(accuracy_score(y_valid, y_pred)*100))
#     # 사이트 에서의 결과와 로컬 결과가 다른건가?
#
    # # 전처리 결과를 다시 DB에 저장 후 불러온 후에 이어서 진행
    # upload_to_db(
    #     yaml_file_path=yaml_file_path,
    #     table_name=f"preprocessed_{table_name}",
    #     train_data=train_data,
    #     test_data=test)

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier
from DataReader import read_from_db
from DataUploader import upload_to_db
import preprocessing
import optuna

def set_initial_setting():
    warnings.filterwarnings(action='ignore')
    plt.style.use("ggplot")
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', 500)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

def preprocess_data(train_data, test, label, target_col):
    train_data, test, label = preprocessing.run(train_data, test, label, target_col, verbose=0)
    return train_data, test, label

def save_submission_file(test_id, y_pred, submission_file_path):
    submission_df = pd.DataFrame({'id': test_id, 'target': y_pred})
    submission_df = submission_df.dropna().drop_duplicates(subset=['id'])
    submission_df.to_csv(submission_file_path, index=False)

def objective(trial):
    # Define hyperparameters to be optimized
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'depth': trial.suggest_int('depth', 1, 10),
        'random_state': 42,
        # 'l2_leaf_reg'
        # 'random_strength'
        # 'bagging_temperature'
        # 'border_count'
        # 'leaf_estimation_method'
        'verbose': 100
    }

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize array to store average validation accuracy
    avg_val_accuracy = []

    # Perform cross validation
    for train_index, test_index in skf.split(train_data, label):
        x_train, x_valid = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_valid = label[train_index], label[test_index]

        # Train model with current hyperparameters
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train, verbose=False)

        # Calculate validation accuracy
        val_accuracy = model.score(x_valid, y_valid)
        avg_val_accuracy.append(val_accuracy)

    # Return average validation accuracy as objective value
    return np.mean(avg_val_accuracy)

if __name__ == "__main__":
    set_initial_setting()

    yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
    data_path = './data/'
    table_name = "adult_income"

    # Read data from PoestgreDB
    train_data, test, label = read_from_db(yaml_file_path=yaml_file_path, table_name=table_name, label_col_name='target')
    test_id = test['id']

    # Preprocess data
    train_data, test, label = preprocess_data(train_data, test, label, 'target')

    # Define study object and optimize hyperparameters
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Get best hyperparameters
    best_params = study.best_params

    # Train final model with best hyperparameters
    model = CatBoostClassifier(**best_params)
    model.fit(train_data, label, verbose=False)

    # Make predictions
    y_pred = model.predict(test)

    # Save submission file
    submission_file = os.path.join(data_path, 'submission.csv')
    save_submission_file(test_id, y_pred, submission_file)


    # # Upload to DB
    # upload_to_db(
    #     yaml_file_path=yaml_file_path,
    #     table_name=f"preprocessed_{table_name}",
    #     train_data=train_data,
    #     test_data=test)