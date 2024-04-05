# 필요한 라이브러리 임포트
import os
import warnings
from datetime import datetime
import itertools
import multiprocessing
import json
import pickle
import sys

import lightgbm as lgb
import xgboost as xbg

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, \
    PowerTransformer, QuantileTransformer
from sklearn.model_selection import StratifiedKFold

import optuna
from sqlalchemy import create_engine, text
import yaml
import pprint

import torch
from torch.utils.data import TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier

# CPU 코어 수 계산
n_cpus = multiprocessing.cpu_count()

# 사용할 작업 스레드 수 설정
n_jobs = max(1, n_cpus - 1)

# 최적의 모델을 담을 변수 초기화
# best_model = None

# 전역 변수 초기화
X_train = None
X_test = None
Y_train = None



class Utils:
    def __init__(self):
        pass

    @staticmethod
    def read_yaml(file_path: str):
        """
        Read YAML file and return data.

        Parameters:
            file_path (str): The path to the YAML file.

        Returns:
            data (dict): The data read from the YAML file.
        """
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data


def read_from_db(yaml_file_path: str, table_name: str, label_col_name: str):
    try:
        # YAML 파일에서 데이터베이스 정보 읽기
        global_info = Utils.read_yaml(yaml_file_path)

        # 데이터베이스 연결
        engine = create_engine(
            url='postgresql+psycopg2://' + global_info['db']['id'] + ':' + global_info['db']['pw'] + '@' +
                global_info['db']['host'] + ':' + global_info['db']['port'] + '/' + global_info['db']['table'])

        with engine.connect() as conn:
            # 트레인 데이터 조회
            sql_for_data = f"SELECT * FROM {table_name}_train"
            data = pd.read_sql(text(sql_for_data), con=conn)

            # 테스트 데이터 조회
            sql_for_test = f"SELECT * FROM {table_name}_test"
            test = pd.read_sql(text(sql_for_test), con=conn)

        # 레이블 추출
        label = data[label_col_name]
        data.drop(columns=[label_col_name], inplace=True)

        return data, test, label
    except Exception as e:
        print(f"An error occurred during reading from the database: {e}")

class PostgreDB:

def postgre_db_engine(db_info):
    db_url = f"postgresql+psycopg2://{db_info['db']['id']}:{db_info['db']['pw']}@" \
             f"{db_info['db']['host']}:{db_info['db']['port']}/{db_info['db']['table']}"
    engine = create_engine(url=db_url)

    return engine

def upload_dataset_to_db(yaml_file_path: str, table_name: str, data_path: str = None,
                 train_df: pd.DataFrame = None, test_df: pd.DataFrame = None):
    try:
        # YAML 파일에서 데이터베이스 정보 읽기
        with open(yaml_file_path, 'r') as file:
            db_info = yaml.safe_load(file)

        # 데이터 경로가 제공되지 않은 경우 예외 발생
        if not data_path and not train_df and not test_df:
            raise ValueError("Please provide the parameter data_path, train_df, or test_df")

        # 데이터베이스 연결 엔진 생성
        engine = postgre_db_engine(db_info)

        # 데이터프레임을 데이터베이스에 업로드
        if data_path:
            train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
            test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

        if train_df is not None:
            train_df.to_sql(f"{table_name}_train", engine, if_exists='append', index=False)
        if test_df is not None:
            test_df.to_sql(f"{table_name}_test", engine, if_exists='append', index=False)

        print("Data uploaded to database successfully.")
    except Exception as e:
        print(f"An error occurred during database upload: {e}")

def upload_model_param_to_db(yaml_file_path: str, table_name: str, model_param, data_path: str = None):
    try:
        # YAML 파일에서 데이터베이스 정보 읽기
        with open(yaml_file_path, 'r') as file:
            global_info = yaml.safe_load(file)

        # 데이터 경로가 제공되지 않은 경우 예외 발생
        if not data_path and not train_df and not test_df:
            raise ValueError("Please provide the parameter data_path, train_df, or test_df")

        # 데이터베이스 연결 엔진 생성
        db_url = f"postgresql+psycopg2://{global_info['db']['id']}:{global_info['db']['pw']}@" \
                 f"{global_info['db']['host']}:{global_info['db']['port']}/{global_info['db']['table']}"
        engine = create_engine(url=db_url)

        # 데이터프레임을 데이터베이스에 업로드
        if data_path:
            train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
            test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

        if train_df is not None:
            train_df.to_sql(f"{table_name}_train", engine, if_exists='append', index=False)
        if test_df is not None:
            test_df.to_sql(f"{table_name}_test", engine, if_exists='append', index=False)

        print("Data uploaded to database successfully.")
    except Exception as e:
        print(f"An error occurred during database upload: {e}")

def optimize_pandas_dtypes(pd_obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    tmp_pd_obj = pd_obj.copy()

    if isinstance(tmp_pd_obj, pd.DataFrame):
        # DataFrame일 경우
        for col in tmp_pd_obj.select_dtypes(include=['float64']).columns:
            tmp_pd_obj[col] = tmp_pd_obj[col].astype('float32')

        for col in tmp_pd_obj.select_dtypes(include=['int64']).columns:
            tmp_pd_obj[col] = tmp_pd_obj[col].astype('int32')

    elif isinstance(tmp_pd_obj, pd.Series):
        # Series일 경우
        if tmp_pd_obj.dtype == 'float64':
            tmp_pd_obj = tmp_pd_obj.astype('float32')

        if tmp_pd_obj.dtype == 'int64':
            tmp_pd_obj = tmp_pd_obj.astype('int32')

    return tmp_pd_obj


# age
def map_age(value):
    if value <= 33:
        return 1
    elif (34 <= value) & (value <= 61):
        return 2
    else:
        return 3


def map_age(value):
    if value <= 33:
        return 1
    elif 34 <= value <= 61:
        return 2
    else:
        return 3


def map_education(value):
    if value <= 8:
        return 1
    elif 9 <= value <= 10:
        return 2
    elif 11 <= value <= 12:
        return 3
    else:
        return 4


def map_workclass(value):
    if value in ['Never-worked', 'Without-pay', 'NAN']:
        return 0
    elif value in ['Private', 'State-gov', 'Self-emp-not-inc', 'Local-gov']:
        return 1
    else:
        return 2


def map_martial(value):
    if value in ['Married-AF-spouse', 'Married-civ-spouse']:
        return 1
    else:
        return 0


def map_country(value):
    if value in ['Iran', 'Yugoslavia', 'India', 'Japan', 'Greece', 'Canada', 'Italy', 'England']:
        return 3
    elif value in ['Philippines', 'Hungary', 'Taiwan', 'France', 'Cambodia', 'Germany', 'Cuba', 'United-States',
                   'South', 'Ireland', 'Hong', 'Ecuador', 'Laos', 'Poland', 'China']:
        return 2
    elif value in ['Thailand', 'Scotland', 'Honduras', 'El-Salvador', 'Peru', 'Trinadad&Tobago']:
        return 1
    elif value in ['Others', 'Puerto-Rico', 'Jamaica', 'Mexico', 'Vietnam', 'Columbia', 'Dominican-Republic', 'Haiti',
                   'Portugal', 'Holand-Netherlands', 'Nicaragua', 'Guatemala', 'Outlying-US(Guam-USVI-etc)']:
        return 0
    else:
        return np.nan


def display_info(obj: object = None, msg: str = ""):
    if obj is not None:
        print(f" ##### {msg} ##### ")
        pprint.pprint(obj)
        print()


def eda_result_print(train_data: pd.DataFrame, test: pd.DataFrame, label: pd.Series, num_columns: list,
                     cat_columns: list, target_col: str):
    # 학습 데이터와 테스트 데이터 정보 출력
    display_info(train_data, "Train Data")
    display_info(test, "Test Data")

    # 학습 데이터 정보 출력
    display_info(train_data, "Train Data loaded")
    display_info(train_data.head(), "Head of Train Data")
    display_info(train_data.shape, "Shape of Train Data")
    display_info(train_data.info(), "Info of Train Data")
    display_info(label.info(), "Info of Label")
    display_info(train_data.describe(include='all'), "Description of Train Data")

    # 테스트 데이터 정보 출력
    display_info(test.shape, "Shape of Test Data")
    display_info(test.info(), "Info of Test Data")
    display_info(test.describe(include='all'), "Description of Test Data")

    # 수치형 변수 정보 출력
    for col in num_columns:
        display_info(train_data[col], col)
        display_info(train_data[col].describe(), col + "'s Description")
        display_info(train_data[col].unique(), col + "'s Unique Values")

    # 범주형 변수 정보 출력
    for col in cat_columns:
        display_info(train_data[col], col)
        display_info(train_data[col].describe(), col + "'s Description")
        ratios = train_data[col].value_counts(normalize=True)
        display_info(ratios, f"Ratio of {col}.")
        display_info(test[col].unique(), col + "'s Unique Values")

    # 결측치 확인
    display_info(cat_columns, "Cat Columns")
    display_info(num_columns, "Num Columns")
    display_info(train_data.isna().sum(), "NaNs in Train Data")
    display_info(test.isna().sum(), "NaNs in Test")

    # 클래스 불균형 확인
    print(f"Target:{target_col} imbalanced 확인")
    target_counts = label.value_counts(normalize=True)
    for value, ratio in target_counts.items():
        print(f"Value: {value}, Ratio: {ratio:.4f}")
    print()


def label_encoder(df: pd.DataFrame, columns: list, le: LabelEncoder) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        label_encoded = le.fit_transform(df[col])
        result[col] = label_encoded
    return result


def one_hot_encoder(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    result = df.copy()
    ohe = OneHotEncoder()

    for col in columns:
        # 범주형 변수에 대해 OneHotEncoder 피팅 및 변환
        one_hot_encoded = ohe.fit_transform(df[[col]])
        ohe_df = pd.DataFrame(one_hot_encoded.toarray(),
                              columns=ohe.get_feature_names_out([col]))

        # 인코딩 결과를 원본 데이터프레임과 결합
        result = pd.concat([result, ohe_df], axis=1)

        # 원본 데이터프레임에서 범주형 변수 삭제
        result.drop(col, axis=1, inplace=True)

    return result


def ordinal_encoder(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    result = df.copy()  # 결과 데이터프레임 생성

    encoder = OrdinalEncoder()  # OrdinalEncoder 객체 생성

    # 범주형 변수에 대해 Ordinal Encoding 수행
    for col in columns:
        result[col] = encoder.fit_transform(df[[col]])  # fit_transform 직접 수행

    return result


def standard_scaler(train: pd.DataFrame, test: pd.DataFrame, columns: list) -> (pd.DataFrame, pd.DataFrame):
    # StandardScaler 객체 생성 및 훈련 데이터에 맞게 적합
    scaler = StandardScaler()
    scaler.fit(train[columns])

    # 훈련 데이터 및 테스트 데이터를 변환
    train_scaled = scaler.transform(train[columns])
    test_scaled = scaler.transform(test[columns])

    # 변환된 데이터를 새로운 데이터프레임으로 생성
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns)
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns)

    # 기존 데이터프레임을 복사하여 수정하지 않고, 스케일링된 데이터를 추가하여 새로운 데이터프레임을 반환
    result_train = train.copy()
    result_test = test.copy()

    result_train = pd.concat([result_train.drop(columns, axis=1), train_scaled_df], axis=1)
    result_test = pd.concat([result_test.drop(columns, axis=1), test_scaled_df], axis=1)

    return result_train, result_test


def numeric_transformer(train, test, columns, strategy='B', viz_available=False, viz_bins=40):
    """
    수치형 특성에 대한 전처리를 수행하는 함수.

    Parameters:
    - train (pd.DataFrame): 훈련 데이터셋
    - test (pd.DataFrame): 테스트 데이터셋
    - columns (list): 전처리를 수행할 특성들의 리스트
    - strategy (str): 전처리 전략 (기본값: 'B' - Box-Cox)
    - viz_available (bool): 시각화 사용 여부 (기본값: False)
    - viz_bins (int): 히스토그램 구간 개수 (기본값: 40)

    Returns:
    - train_result (pd.DataFrame): 전처리된 훈련 데이터셋
    - test_result (pd.DataFrame): 전처리된 테스트 데이터셋
    """
    train_result = train.copy()
    test_result = test.copy()

    # 'id' 컬럼 제거
    train_result_id = train_result.pop('id')
    test_result_id = test_result.pop('id')

    try:
        if strategy in ['L', 'B']:  # Log 또는 Box-Cox
            transformer = PowerTransformer(method='box-cox')
        elif strategy == 'Y':  # Yeo-Johnson
            transformer = PowerTransformer(method='yeo-johnson')
        elif strategy == 'Q':  # Quantile
            transformer = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError("Invalid strategy value. Please choose from 'L', 'B', 'Y', or 'Q'.")

        for col in columns:
            # 전처리 적용
            train_result[col] = transformer.fit_transform(train_result[col].values.reshape(-1, 1))
            test_result[col] = transformer.transform(test_result[col].values.reshape(-1, 1))

            # 시각화
            if viz_available:
                train_result[col].hist(bins=viz_bins)

    except ValueError as e:
        print(f"ValueError: {e}")
        print("Automatically changing strategy to 'Quantile'.")

        for col in columns:
            transformer = QuantileTransformer(output_distribution='normal')
            train_result[col] = transformer.fit_transform(train_result[col].values.reshape(-1, 1))
            test_result[col] = transformer.transform(test_result[col].values.reshape(-1, 1))

            if viz_available:
                train_result[col].hist(bins=viz_bins)

    # 'id' 컬럼 다시 추가
    train_result['id'] = train_result_id
    test_result['id'] = test_result_id

    return train_result, test_result


def control_imbalance(train_data: pd.DataFrame, label: pd.Series, seed: int = 42):
    smote = SMOTE(random_state=seed)
    resampled_data, resampled_label = smote.fit_resample(train_data, label)
    return pd.DataFrame(resampled_data, columns=train_data.columns), pd.Series(resampled_label)


def run_encoding(df: pd.DataFrame, drop_col: list) -> pd.DataFrame:
    # 원래의 데이터프레임을 직접 수정하고 필요한 경우 복사
    encoded_data = df.drop(drop_col, axis=1).reset_index(drop=True)

    # 인코딩할 컬럼들을 변수로 사용
    label_encode_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'native_country']
    one_hot_encode_cols = ['sex']
    ordinal_encode_cols = ['education']

    # label encoding
    encoded_data = label_encoder(df=encoded_data, columns=label_encode_cols)

    # one-hot encoding
    encoded_data = one_hot_encoder(df=encoded_data, columns=one_hot_encode_cols)

    # ordinal encoding
    encoded_data = ordinal_encoder(df=encoded_data, columns=ordinal_encode_cols)

    return encoded_data


def separate_columns(train_data):
    """Separate categorical and numerical columns."""
    cat_columns = train_data.select_dtypes(include='object').columns
    num_columns = train_data.select_dtypes(exclude='object').columns
    return cat_columns, num_columns


def apply_chi2_analysis(train_data, target_col, verbose: int = 0):
    """Apply Chi-squared analysis to column combinations."""
    column_combinations = list(itertools.combinations(target_col, 2))
    if verbose == 1:
        display_info(column_combinations, "column_combinations")

        for combination in column_combinations:
            cross_tab = pd.crosstab(train_data[combination[0]], train_data[combination[1]])
            chi2, p, _, _ = chi2_contingency(cross_tab)
            display_info(chi2, f"chi2 for columns {combination}")
            display_info(p, f"P-value for columns {combination}")

            if verbose == 3:
                sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g')
                plt.title('Cross Tabulation Heatmap')
                plt.show()


def preprocess(train_data: pd.DataFrame, valid_data: pd.DataFrame, test_data: pd.DataFrame, label: pd.Series,
               verbose: int = 0):
    tmp_train_data, tmp_valid_data, tmp_test_data, y_train = None, None, None, label
    # 범주형, 수치형 변수 분리
    cat_columns = train_data.select_dtypes(include='object').columns
    num_columns = train_data.select_dtypes(exclude='object').columns

    if verbose == 2:
        eda_result_print(train_data, test, label, num_columns, cat_columns)

    # 고려할 컬럼
    target_cols = ['workclass', 'occupation', 'native_country', 'race']
    column_combinations = list(itertools.combinations(target_cols, 2))
    if verbose == 1:
        display_info(obj=column_combinations, msg="column_combinations")

    if verbose == 1:
        for combination in column_combinations:
            cross_tab = pd.crosstab(train_data[combination[0]], train_data[combination[1]])
            chi2, p, _, _ = chi2_contingency(cross_tab)
            # display_info(cross_tab, f"Cross tabulation for columns {combination}")
            display_info(obj=chi2, msg=f"chi2 for columns {combination}")
            display_info(obj=p, msg=f"P-value for columns {combination}")

            # 시각화
            if verbose == 3:
                sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g')
                plt.title('Cross Tabulation Heatmap')
                plt.show()

    # 컬럼명의 '.' 문자 변경
    tmp_train_data = train_data.copy()
    try:
        tmp_valid_data = valid_data.copy()
    except:
        pass
    tmp_test_data = test_data.copy()

    tmp_train_data.reset_index(drop=True, inplace=True)
    try:
        tmp_valid_data.reset_index(drop=True, inplace=True)
    except:
        pass
    tmp_test_data.reset_index(drop=True, inplace=True)

    # age
    tmp_train_data['age'] = tmp_train_data['age'].apply(map_age)
    try:
        tmp_valid_data['age'] = tmp_valid_data['age'].apply(map_age)
    except:
        pass
    tmp_test_data['age'] = tmp_test_data['age'].apply(map_age)

    # education.num / [1~8 / 9~10 / 11~12 / 13~16]
    tmp_train_data['education_num'] = tmp_train_data['education_num'].apply(map_education)
    try:
        tmp_valid_data['education_num'] = tmp_valid_data['education_num'].apply(map_education)
    except:
        pass
    tmp_test_data['education_num'] = tmp_test_data['education_num'].apply(map_education)

    # marital.status
    tmp_train_data['marital_status'] = tmp_train_data['marital_status'].apply(map_martial)
    try:
        tmp_valid_data['marital_status'] = tmp_valid_data['marital_status'].apply(map_martial)
    except:
        pass
    tmp_test_data['marital_status'] = tmp_test_data['marital_status'].apply(map_martial)

    # education(drop)
    tmp_train_data.drop(columns='education', inplace=True)
    try:
        tmp_valid_data.drop(columns='education', inplace=True)
    except:
        pass
    tmp_test_data.drop(columns='education', inplace=True)

    # capital.gain
    trans = QuantileTransformer(output_distribution='normal')
    trans.fit(tmp_train_data[tmp_train_data['capital_gain'] < 20000]['capital_gain'].values.reshape(-1, 1))

    tmp_train_data['capital_gain'] = trans.transform(tmp_train_data['capital_gain'].values.reshape(-1, 1))
    try:
        tmp_valid_data['capital_gain'] = trans.transform(tmp_valid_data['capital_gain'].values.reshape(-1, 1))
    except:
        pass
    tmp_test_data['capital_gain'] = trans.transform(tmp_test_data['capital_gain'].values.reshape(-1, 1))
    # valid, test에 대한 transform

    # capital.loss
    trans = QuantileTransformer(output_distribution='normal')
    trans.fit(tmp_train_data[tmp_train_data['capital_loss'] < 3000]['capital_loss'].values.reshape(-1, 1))

    tmp_train_data['capital_loss'] = trans.transform(tmp_train_data['capital_loss'].values.reshape(-1, 1))
    try:
        tmp_valid_data['capital_loss'] = trans.transform(tmp_valid_data['capital_loss'].values.reshape(-1, 1))
    except:
        pass
    tmp_test_data['capital_loss'] = trans.transform(tmp_test_data['capital_loss'].values.reshape(-1, 1))

    # hours
    trans = QuantileTransformer(output_distribution='normal')
    trans.fit(tmp_train_data['hours_per_week'].values.reshape(-1, 1))

    tmp_train_data['hours_per_week'] = trans.transform(tmp_train_data['hours_per_week'].values.reshape(-1, 1))
    try:
        tmp_valid_data['hours_per_week'] = trans.transform(tmp_valid_data['hours_per_week'].values.reshape(-1, 1))
    except:
        pass
    tmp_test_data['hours_per_week'] = trans.transform(tmp_test_data['hours_per_week'].values.reshape(-1, 1))

    # normalize
    scaler = MinMaxScaler()  # 이상치에 더 강한 양상을 보이는 scaler 
    num_col = ['capital_gain', 'capital_loss', 'hours_per_week']

    tmp_train_data[num_col] = scaler.fit_transform(tmp_train_data[num_col])
    try:
        tmp_valid_data[num_col] = scaler.transform(tmp_valid_data[num_col])
    except:
        pass
    tmp_test_data[num_col] = scaler.transform(tmp_test_data[num_col])

    # 결측치 처리: 결측치는 Train 데이터에만 존재함
    # native_country 결측치 처리
    tmp_train_data.loc[(tmp_train_data['race'] == 'White') & (
        tmp_train_data['native_country'].isin([np.nan, None, 'None'])), 'native_country'] = 'United-States'
    tmp_train_data.loc[tmp_train_data['race'] != 'White', 'native_country'] = 'Others'

    # native.country
    tmp_train_data['native_country'] = tmp_train_data['native_country'].apply(map_country)
    try:
        tmp_valid_data['native_country'] = tmp_valid_data['native_country'].apply(map_country)
    except:
        pass
    tmp_test_data['native_country'] = tmp_test_data['native_country'].apply(map_country)

    # # workclass 결측치 처리
    tmp_train_data['workclass'] = tmp_train_data['workclass'].fillna('NAN')
    tmp_train_data.loc[tmp_train_data['workclass'].isin([np.nan, None, 'None']), 'workclass'] = 'NAN'
    try:
        tmp_valid_data['workclass'] = tmp_valid_data['workclass'].fillna('NAN')
        tmp_valid_data.loc[tmp_valid_data['workclass'].isin([np.nan, None, 'None']), 'workclass'] = 'NAN'
    except:
        pass
    tmp_test_data['workclass'] = tmp_test_data['workclass'].fillna('NAN')
    tmp_test_data.loc[tmp_test_data['workclass'].isin([np.nan, None, 'None']), 'workclass'] = 'NAN'

    # workclass
    tmp_train_data['workclass'] = tmp_train_data['workclass'].apply(map_workclass)
    try:
        tmp_valid_data['workclass'] = tmp_valid_data['workclass'].apply(map_workclass)
    except:
        pass
    tmp_test_data['workclass'] = tmp_test_data['workclass'].apply(map_workclass)

    # occupation 결측치 처리
    tmp_train_data['occupation'] = tmp_train_data['occupation'].fillna('NAN')
    tmp_train_data.loc[tmp_train_data['occupation'].isin([np.nan, None, 'None']), 'occupation'] = 'NAN'

    # 결과 확인
    if verbose == 1:
        display_info(obj=pd.isna(train_data).sum(), msg="결측치 처리 결과")

    drop_col = 'fnlwgt'
    try:
        tmp_train_data.drop(drop_col, axis=1, inplace=True)
        try:
            tmp_valid_data.drop(drop_col, axis=1, inplace=True)
        except:
            pass
        tmp_test_data.drop(drop_col, axis=1, inplace=True)
        num_columns = num_columns.drop(drop_col)
    except KeyError:
        print(f'Warning: [{drop_col}] not found in axis. Maybe I think [{drop_col}] is already dropped."')

    # 범주형 변수(OHE)
    cat_col = ['occupation', 'relationship', 'race', 'sex']
    ''' ordinal 인코딩 된 것들과 혼선을 방지하기 위해 원핫 인코딩 처리 '''
    ohe = OneHotEncoder()
    try:
        data_df = pd.concat([
            tmp_train_data[cat_col],
            tmp_valid_data[cat_col],
            tmp_test_data[cat_col]])
    except:
        data_df = pd.concat([
            tmp_train_data[cat_col],
            # tmp_valid_data[cat_col],
            tmp_test_data[cat_col]])
    ohe.fit(data_df)
    ohe_columns = ohe.get_feature_names_out(cat_col)

    new_train_cat = pd.DataFrame(
        ohe.transform(tmp_train_data[cat_col]).toarray(),
        columns=ohe_columns,
        index=tmp_train_data.index
    )
    try:
        new_valid_cat = pd.DataFrame(
            ohe.transform(tmp_valid_data[cat_col]).toarray(),
            columns=ohe_columns,
            index=tmp_valid_data.index
        )
    except:
        pass
    new_test_cat = pd.DataFrame(
        ohe.transform(tmp_test_data[cat_col]).toarray(),
        columns=ohe_columns,
        index=tmp_test_data.index
    )

    tmp_train_data.drop(columns=cat_col, inplace=True)
    try:
        tmp_valid_data.drop(columns=cat_col, inplace=True)
    except:
        pass
    tmp_test_data.drop(columns=cat_col, inplace=True)

    tmp_train_data = pd.concat([tmp_train_data, new_train_cat], axis=1)
    try:
        tmp_valid_data = pd.concat([tmp_valid_data, new_valid_cat], axis=1)
    except:
        pass
    tmp_test_data = pd.concat([tmp_test_data, new_test_cat], axis=1)

    tmp_train_data.drop('index', axis=1, inplace=True)
    try:
        tmp_valid_data.drop('index', axis=1, inplace=True)
    except:
        pass
    tmp_test_data.drop('index', axis=1, inplace=True)

    tmp_train_data.drop('id', axis=1, inplace=True)
    try:
        tmp_valid_data.drop('id', axis=1, inplace=True)
    except:
        pass
    tmp_test_data.drop('id', axis=1, inplace=True)

    # train_data, test = numeric_transformer(train=train_data, test=test, columns=num_columns, strategy='Q', viz_available=True)
    # train_data, test = standard_scaler(train=train_data, test=test, columns=num_columns)

    # tmp_train_data, y_train = control_imbalance(train_data=tmp_train_data, label=label)

    # pandas dtype 최적화
    tmp_train_data = optimize_pandas_dtypes(pd_obj=tmp_train_data)
    try:
        tmp_valid_data = optimize_pandas_dtypes(pd_obj=tmp_valid_data)
    except:
        pass
    tmp_test_data = optimize_pandas_dtypes(pd_obj=tmp_test_data)
    y_train = optimize_pandas_dtypes(pd_obj=y_train)

    return tmp_train_data, tmp_valid_data, tmp_test_data, y_train


def set_initial_setting(base_dir: str, folder_names: list):
    warnings.filterwarnings(action='ignore')
    plt.style.use("ggplot")
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', 500)
    np.random.seed(42)
    os.environ['PYTHONHASHSEED'] = str(42)

    if torch.cuda.is_available():
        device = 'cuda'
        print("GPU 사용 가능")
    else:
        device = 'cpu'
        print("GPU 사용 불가, CPU 사용")

    if base_dir and folder_names:
        for name in folder_names:
            folder_path = os.path.join(base_dir, name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"'{name}' 폴더가 생성되었습니다.")

    if env == "colab":
        from google.colab import drive
        drive.mount('/content/drive')


def save_submission_file(model_name, test_id, y_pred, submission_file_path) -> pd.DataFrame:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)

    submission_df = pd.DataFrame({'id': test_id, 'target': y_pred})
    submission_df = submission_df.dropna().drop_duplicates(subset=['id'])
    submission_df.to_csv(f"{submission_file_path}submission-{model_name}-{current_time}.csv", index=False)

    return submission_df


def save_model(model_type: str, model, file_path: str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if model_type == 'catboost':
        model.save_model(
            fname=f"{file_path}model-{model_type}-{current_time}.cbm",
            format="cbm"  # todo: "onnx"
        )
    elif model_type == 'xgboost':
        model.save_model(f"{file_path}model-{model_type}-{current_time}.json")
    elif model_type == 'lightgbm':
        model.booster_.save_model(f"{file_path}model-{model_type}-{current_time}.txt")  # 텍스트 파일 형식으로 모델 저장
    elif model_type == 'tabnet':
        # Assuming `file_path` ends with a slash
        model_path = f"{file_path}model-{model_type}-{current_time}.zip"
        model.save_model(model_path)
    else:
        with open(f"{file_path}model-{model_type}-{current_time}.pkl", 'wb') as f:
            pickle.dump(model.state_dict(), f)


def save_parameters(model_name: str, best_params, file_path: str):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(f"{file_path}model-parameters-{model_name}-{current_time}", 'w') as f:
        json.dump(best_params, f, indent=4)


def save_best_params(model_name: str, model, file_path):
    if model_name == 'catboost':
        best_params = model.get_all_params()
    elif model_name == 'xgboost':
        best_params = model.get_booster().attributes()
    elif model_name == 'lightgbm':
        best_params = model.get_params()
    elif model_name == 'tabnet':
        best_params = model.get_params()
    save_parameters(model_name, best_params, file_path=f"{file_path}")
    return best_params


def create_model(model_name, params, n_jobs):
    if model_name == 'catboost':
        model = CatBoostClassifier(**params, thread_count=n_jobs)
    elif model_name == 'xgboost':
        model = XGBClassifier(**params, n_jobs=n_jobs)
    elif model_name == 'lightgbm':
        model = LGBMClassifier(**params, n_jobs=n_jobs)
    elif model_name == 'tabnet':
        model = TabNetClassifier(
            **params,
            device_name='cuda' if torch.cuda.is_available() else 'cpu'  # Automatically use GPU if available
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model


def train_model(model_name, model, x_train: pd.DataFrame, y_train: pd.DataFrame, x_valid: pd.DataFrame = None,
                y_valid: pd.DataFrame = None, verbose: bool = False,
                early_stopping_rounds: int = 100):
    if model_name == 'catboost':
        model.fit(
            x_train,
            y_train,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds
        )
    elif model_name == 'xgboost':
        model.fit(
            x_train, y_train,
            verbose=verbose
        )
    elif model_name == 'lightgbm':
        model.fit(
            x_train, y_train,
        )
    elif model_name == 'tabnet':
        max_epochs = model.max_epochs if hasattr(model, 'max_epochs') else 100

        model.fit(
            X_train=x_train.values, y_train=y_train.values,
            max_epochs=max_epochs,
            patience=early_stopping_rounds,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=n_jobs,
            drop_last=False
        )
    return model


def train_and_evaluate(model_name: str, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                       y_test: pd.Series,
                       params: dict, n_jobs: int, verbose: bool = False, early_stopping_rounds: int = 100,
                       extract_best_model: bool = False):
    """
    Train and evaluate a machine learning model.

    Args:
    - X_train (pd.DataFrame): Training features.
    - X_test (pd.DataFrame): Test features.
    - y_train (pd.Series): Training labels.
    - y_test (pd.Series): Test labels.
    - params (dict): Parameters for creating the model.
    - n_jobs (int): Number of jobs to run in parallel.
    - verbose (bool, optional): Whether to print progress messages. Default is False.
    - early_stopping_rounds (int, optional): Early stopping rounds for training. Default is 100.
    - extract_best_model (bool, optional): Whether to extract the best model. Default is False.

    Returns:
    - float: Mean validation accuracy.
    """
    model = create_model(model_name=model_name, params=params, n_jobs=n_jobs)
    model = train_model(model_name, model, x_train, y_train, x_test, y_test, verbose=verbose,
                        early_stopping_rounds=early_stopping_rounds)

    val_accuracy = model.score(x_test, y_test)

    return np.mean(val_accuracy)


def train_tabnet(train_data, label, test, params, n_jobs=1, verbose=False, early_stopping_rounds=None,
                 extract_best_model=False):
    # TabNet 모델 초기화
    tabnet_model = TabNetClassifier(**params)

    # GPU 사용 가능 여부 확인
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # 모델을 GPU 또는 CPU에 할당
    tabnet_model.device = device

    # 모델 학습
    tabnet_model.fit(
        X_train=x_train,
        y_train=y_train,
        # eval_name=['valid'],
        # eval_metric=['accuracy'],  # 원하는 평가 지표로 변경 가능
        max_epochs=params['max_epochs'],
        patience=early_stopping_rounds,  # 조기 종료를 위한 패턴스 설정
        batch_size=params['virtual_batch_size'],  # 가상 배치 사이즈 설정
        num_workers=n_jobs,
        drop_last=False,
        loss_fn=torch.nn.functional.cross_entropy,  # TabNet은 다중 분류를 위한 cross entropy loss를 사용
        # verbose=verbose
    )

    # 최적 모델 추출
    if extract_best_model:
        tabnet_model.load_best_model()

    return tabnet_model


def objective(trial, model_name):
    # model_name = trial.suggest_categorical('model_name', ['catboost', 'xgboost', 'lightgbm'])

    if model_name != 'tabnet':
        # Initialize StratifiedKFold
        skf = StratifiedKFold(
            n_splits=5,  # todo: n_splits
            shuffle=True,
            random_state=42
        )

    # Define hyperparameters to be optimized
    if env == "colab":
        if model_name == 'catboost':
            params = {  # todo: hyperparameter range
                'iterations': trial.suggest_int('iterations', 300, 1000),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                'depth': trial.suggest_int('depth', 2, 10),
                'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.1, 10.0),
                'random_strength': trial.suggest_uniform('random_strength', 0.0, 1.0),
                'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
                'border_count': trial.suggest_int('border_count', 50, 255),
                'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
                'verbose': 100,
                'task_type': 'GPU',
                'devices': '0',
                'scale_pos_weight': 3.25,
                # 'eval_metrics': 'Accuracy',  # Ref: https://catboost.ai/en/docs/references/custom-metric__supported-metrics
            }
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'scale_pos_weight': 3.25,
                'tree_method': 'gpu_hist',
            }
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', -1, 16),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
                'sample_weight': 3.25,
            }
        elif model_name == "tabnet":
            params = {
                'n_d': trial.suggest_int('n_d', 8, 64),
                'n_a': trial.suggest_int('n_a', 8, 64),
                'max_epochs': trial.suggest_int('max_epochs', 10, 100),
                'virtual_batch_size': trial.suggest_int('virtual_batch_size', 32, 128),
                # Other TabNet-specific parameters
            }
    elif env == "local":
        if model_name == 'catboost':
            params = {
                'iterations': 1,
                'learning_rate': 0.03,
                'scale_pos_weight': 3.25,
            }
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1, 10),
                'max_depth': 3,
            }
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': 3,
                'max_depth': 3,
            }
        elif model_name == "tabnet":
            params = {
                'n_d': 2,
                'n_a': 2,
                'max_epochs': 3
            }

    avg_val_accuracy = []

    global X_train
    global X_test
    global Y_train

    if model_name != 'tabnet':
        for i, (trn_idx, val_idx) in enumerate(skf.split(train_data, label)):
            x_train, y_train = train_data.iloc[trn_idx], label[trn_idx]
            x_valid, y_valid = train_data.iloc[val_idx], label[val_idx]

            # 전처리
            x_train, x_valid, x_test, y_train = preprocess(x_train, x_valid, test, y_train, verbose=0)

            X_train = x_train
            X_test = x_test
            Y_train = y_train

            val_accuracy = train_and_evaluate(model_name, x_train, y_train, x_valid, y_valid, params, n_jobs=n_jobs,
                                              verbose=False,
                                              early_stopping_rounds=100, extract_best_model=True)
            avg_val_accuracy.append(val_accuracy)

            # 검증 데이터 메트릭
            # val_auc = roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1])
            # auc_value += (val_auc/5)

        return np.mean(avg_val_accuracy)
    else:
        data = pd.DataFrame(np.random.randn(262200, 10), columns=[f'feature_{i}' for i in range(10)])

        # DataFrame의 행 인덱스를 리스트로 가져오기
        indices = list(data.index)

        # 행 인덱스를 무작위로 섞기
        np.random.shuffle(indices)

        # 섞인 인덱스를 사용하여 DataFrame 재구성
        shuffled_data = data.iloc[indices]

        x_train_indices = train_data.index[:int(0.8 * len(train_data))]
        y_train_indices = label.index[:int(0.8 * len(train_data))]

        # 나머지 0.2개를 테스트 데이터로 설정
        x_valid_indices = train_data.index[int(0.8 * len(train_data)):]
        y_valid_indices = label.index[int(0.8 * len(train_data)):]

        x_train = train_data.loc[x_train_indices]
        y_train = label.loc[y_train_indices]
        x_valid = train_data.loc[x_valid_indices]
        y_valid = label.loc[y_valid_indices]

        x_train, x_valid, x_test, y_train = preprocess(x_train, x_valid, test, y_train, verbose=0)
        # global X_train
        # global X_test
        # global Y_train
        X_train = x_train
        X_test = x_test
        Y_train = y_train

        # TabNet 모델 학습
        val_accuracy = train_and_evaluate(model_name, x_train, y_train, x_valid, y_valid, params, n_jobs=n_jobs,
                                          verbose=False,
                                          early_stopping_rounds=100, extract_best_model=True)
        # avg_val_accuracy.append(val_accuracy)

        return np.array(val_accuracy)


if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python <script.py> <model_name>")
    #     sys.exit(1)
    #
    # model_name = sys.argv[1].lower()

    # 실행 환경에 따른 설정(필수)
    # env = "colab"
    env = "local"

    # 모델 선택
    model_name = 'catboost'
    # model_name = 'xgboost'
    # model_name = 'lightgbm' # GPU 지원 X
    # model_name = 'tabnet' # 사용 X

    # tunning = True
    tunning = False

    if model_name not in ['catboost', 'xgboost', 'lightgbm', 'tabnet']:
        print("Model name must be one of 'catboost', 'xgboost', 'lightgbm', 'tabnet'")
        sys.exit(1)

    if env == "colab":
        yaml_file_path = '/content/drive/MyDrive/info.yaml'
        base_path = '/content/drive/MyDrive'
        data_path = f'{base_path}/Adult_Income/'
    elif env == "local":
        yaml_file_path = './info.yaml'
        base_path = '.'
        data_path = f'{base_path}/data/'

    set_initial_setting(base_dir=base_path, folder_names=['log', 'model', 'submission'])
    model_path = '/model/'
    table_name = "adult_income"

    # Read data from PoestgreDB
    train_data, test, label = read_from_db(
        yaml_file_path=yaml_file_path,
        table_name=table_name,
        label_col_name='target'
    )

    train_data = optimize_pandas_dtypes(pd_obj=train_data)
    test = optimize_pandas_dtypes(pd_obj=test)
    label = optimize_pandas_dtypes(pd_obj=label)

    test_id = test['id']

    train_data.columns = train_data.columns.str.replace('.', '_')
    test.columns = test.columns.str.replace('.', '_')

    cat_columns = train_data.select_dtypes(include='object').columns
    num_columns = train_data.select_dtypes(exclude='object').columns

    if tunning:
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(lambda trial: objective(trial, model_name), n_trials=100)  # todo

        best_params = study.best_params

    x_train, _, x_test, y_train = preprocess(train_data, None, test, label, verbose=0)

    X_train = x_train
    X_test = x_test
    Y_train = y_train

    best_params = {'task_type': 'GPU', 'devices': '0', 'iterations': 937, 'learning_rate': 0.08792393842837028,
                   'depth': 10, 'l2_leaf_reg': 0.4963956948794404, 'random_strength': 0.15890188094827148,
                   'bagging_temperature': 0.07307433884939663, 'border_count': 224, 'leaf_estimation_method': 'Newton'}
    best_model = create_model(model_name=model_name, params=best_params, n_jobs=n_jobs)
    best_trained_model = train_model(model_name, best_model, X_train, Y_train)

    if tunning:
        save_best_params(model_name=model_name, model=best_trained_model, file_path=f"{base_path}/model/")

        save_model(
            model_type=model_name,
            model=best_trained_model,
            file_path=f"{base_path}/model/"
        )

    y_pred = best_trained_model.predict_proba(X_test)
    y_pred_argmax = np.argmax(y_pred, axis=1)

    save_submission_file(
        model_name=model_name,
        test_id=test_id,
        y_pred=y_pred_argmax,
        submission_file_path=f'{base_path}/submission/'
    )



    print("Finished.")