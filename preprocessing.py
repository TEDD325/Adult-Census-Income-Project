import os
import warnings
from scipy.stats import chi2_contingency
import itertools
import pprint
import DataReader
import DataUploader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

def set_initial_setting():
    warnings.filterwarnings(action='ignore')
    plt.style.use("ggplot")
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', 500)

    # seed 설정
    seed_num = 42
    np.random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)

def display_info(obj: object = None, msg: str = ""):
    # print()
    pprint.pp(" ##### " + msg + " ##### ")
    pprint.pp(obj)
    print(end="\n\n")


def eda_result_print(train_data: pd.DataFrame, test: pd.DataFrame, label: pd.Series, num_columns: list, cat_columns: list, target_col: str):
    display_info(train_data, "Train Data loaded")
    display_info(train_data.head(), "Head of Train Data")
    display_info(train_data.shape, "Shape of Train Data")  # (17480, 17)
    display_info(test.shape, "Shape of Test Data")  # (15081, 16)
    display_info(train_data.info(), "Info of Train Data")
    display_info(test.info(), "Info of Test Data")
    display_info(label.info(), "Info of Label")
    display_info(train_data.describe(include='all'), "Description of Train Data")
    display_info(test.describe(include='all'), "Description of Test Data")

    for col in num_columns:
        display_info(train_data[col], col)
        display_info(train_data[col].describe(), col + "'s Description")
        display_info(train_data[col].unique(), col + "'s Unique Values")

    for col in cat_columns:
        display_info(train_data[col], col)
        display_info(test[col].describe(), col + "'s Description")
        display_info(test[col].unique(), col + "'s Unique Values")

    # 결측치 확인
    display_info(cat_columns, "Cat Columns")
    display_info(num_columns, "Num Columns")
    display_info(train_data.isna().sum(), "NaNs in Train Data")
    display_info(test.isna().sum(), "NaNs in Test")

    # 클래스 imbalanced 확인
    for idx, data in enumerate([train_data]):
        if idx==0: print("Train imbalanced 확인")

        tmp_data = pd.concat([data, label], axis=1)

        total_count = len(tmp_data)
        unique_values = tmp_data[target_col].unique()
        for value in unique_values:
            count = (tmp_data[target_col] == value).sum()
            ratio = count / total_count
            print(f"Value: {value}, Count: {count}, Ratio: {ratio:.4f}")
        print()
    print(end="\n\n")


def run(train_data: pd.DataFrame, test: pd.DataFrame, label: pd.Series, target_col: str, verbose: int = 0):
    # yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
    # target_col = 'target'

    # set_initial_setting()

    # # 데이터 읽기
    # train_data, test, label = DataReader.read_from_db(
    #     yaml_file_path=yaml_file_path,
    #     table_name="adult_income",
    #     label_col_name=target_col)

    # 컬럼명의 '.' 문자 변경
    train_data.columns = train_data.columns.str.replace('.', '_')
    test.columns = test.columns.str.replace('.', '_')

    # 범주형, 수치형 변수 분리
    cat_columns = train_data.select_dtypes(include='object').columns
    num_columns = train_data.select_dtypes(exclude='object').columns

    if verbose == 2:
        eda_result_print(train_data, test, label, num_columns, cat_columns)

    # 고려할 컬럼
    # missing_col = ['workclass', 'occupation', 'native_country', 'race']
    missing_cols = train_data.columns[train_data.isna().any()].tolist()
    column_combinations = list(itertools.combinations(missing_cols, 2))
    # column_combinations = list(itertools.combinations(cat_columns, 2))
    display_info(column_combinations, "column_combinations")

    for combination in column_combinations:
        cross_tab = pd.crosstab(train_data[combination[0]], train_data[combination[1]])
        chi2, p, _, _ = chi2_contingency(cross_tab)
        # display_info(cross_tab, f"Cross tabulation for columns {combination}")
        display_info(chi2, f"chi2 for columns {combination}")
        display_info(p, f"P-value for columns {combination}")

        # # 시각화
        if verbose == 3:
            sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g')
            plt.title('Cross Tabulation Heatmap')
            plt.show()

    # 결측치 처리
    # native_country 결측치 처리
    tmp_train_data = train_data.copy()
    tmp_train_data.loc[(tmp_train_data['race'] == 'White') & (tmp_train_data['native_country'].isin([np.nan, None, 'None'])), 'native_country'] = 'United-States'
    tmp_train_data.loc[tmp_train_data['race'] != 'White', 'native_country'] = 'Others'

    # workclass 결측치 처리
    tmp_train_data['workclass'] = tmp_train_data['workclass'].fillna('Never-worked')

    # occupation 결측치 처리
    tmp_train_data['occupation'] = tmp_train_data['occupation'].fillna('Others')

    # 결과 확인
    display_info(pd.isna(tmp_train_data).sum(), "결측치 처리 결과")

    train_data = tmp_train_data

    drop_col = 'fnlwgt'
    train_data.drop(drop_col, axis=1, inplace=True)
    test.drop(drop_col, axis=1, inplace=True)
    num_columns = num_columns.drop(drop_col)

    if verbose == 1:
        eda_result_print(train_data, test, label, num_columns, cat_columns, target_col)
    # 레이블이 0인 데이터가 레이블이 1인 데이터보다 약 3배 정도 많다.
    '''
    imbalanced 처리 방법
    
    SMOTE(Synthetic Minority Over-sampling Technique)
    클래스 가중치 조정
    AdaBoost나 Gradient Boosting 등의 앙상블 기법
    정밀도(precision), 재현율(recall), F1 점수 등의 평가 지표를 사용
    
    여러 기법을 조합하여 사용하는 것이 효과적
    '''

    # ENCODING
    display_info(msg='ENCODING')
    # categorical column들 중 어느 컬럼에 어느 방법을 적용해야 하는가?
    '''Index(['workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'native_country'],
      dtype='object')'''
    ''' 
    - Label: 단순 인코딩; 0, 1, 2
    - One-Hot: 0, 1 값만을 가진 새로운 차원 생성
    , Ordinal: 순서가 중요한 경우 '''




    # SMOTE를 적용하려면 인코딩이 끝나야 한다.
    # # SMOTE 객체 생성
    # smote = SMOTE(random_state=42)
    #
    # # SMOTE를 적용할 데이터 준비
    # resampled_train, resampled_label = smote.fit_resample(train_data, label)
    #
    # # 샘플링된 데이터로 데이터프레임 생성 (예시)
    # resampled_train = pd.DataFrame(resampled_train, columns=train_data.columns)
    # # resampled_df[target_col] = resampled_label

    print("Debugging Point")



