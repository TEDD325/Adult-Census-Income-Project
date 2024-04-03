# import os
# import warnings
#
# from pandas import DataFrame
# from scipy.stats import chi2_contingency
# import itertools
# import pprint
# import DataReader
# import DataUploader
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer, QuantileTransformer
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
# def display_info(obj: object = None, msg: str = ""):
#     # print()
#     pprint.pp(" ##### " + msg + " ##### ")
#     pprint.pp(obj)
#     print(end="\n\n")
#
#
# def eda_result_print(train_data: pd.DataFrame, test: pd.DataFrame, label: pd.Series, num_columns: list, cat_columns: list, target_col: str):
#     display_info(obj=train_data, msg="Train Data")
#     display_info(obj=test, msg="Test Data")
#
#     display_info(obj=train_data, msg="Train Data loaded")
#     display_info(obj=train_data.head(), msg="Head of Train Data")
#     display_info(obj=train_data.shape, msg="Shape of Train Data")  # (17480, 17)
#     display_info(obj=test.shape, msg="Shape of Test Data")  # (15081, 16)
#     display_info(obj=train_data.info(), msg="Info of Train Data")
#     display_info(obj=test.info(), msg="Info of Test Data")
#     display_info(obj=label.info(), msg="Info of Label")
#     display_info(obj=train_data.describe(include='all'), msg="Description of Train Data")
#     display_info(obj=test.describe(include='all'), msg="Description of Test Data")
#
#     for col in num_columns:
#         display_info(obj=train_data[col], msg=col)
#         display_info(obj=train_data[col].describe(), msg=col + "'s Description")
#         display_info(obj=train_data[col].unique(), msg=col + "'s Unique Values")
#
#     for col in cat_columns:
#         display_info(obj=train_data[col], msg=col)
#         tmp_df = train_data[col].describe()
#         display_info(obj=tmp_df, msg=col + "'s Description")
#         ratios = train_data[col].value_counts() / len(train_data[col])
#         display_info(obj=ratios, msg=f"Ratio of {col}.")
#         display_info(obj=test[col].unique(), msg=col + "'s Unique Values")
#
#     # 결측치 확인
#     display_info(cat_columns, "Cat Columns")
#     display_info(num_columns, "Num Columns")
#     display_info(train_data.isna().sum(), "NaNs in Train Data")
#     display_info(test.isna().sum(), "NaNs in Test")
#
#     # 클래스 imbalanced 확인
#     for idx, data in enumerate([train_data]):
#         if idx==0: print(f"Target:{target_col} imbalanced 확인")
#
#         tmp_data = pd.concat([data, label], axis=1)
#
#         total_count = len(tmp_data)
#         unique_values = tmp_data[target_col].unique()
#         for value in unique_values:
#             count = (tmp_data[target_col] == value).sum()
#             ratio = count / total_count
#             print(f"Value: {value}, Count: {count}, Ratio: {ratio:.4f}")
#         print()
#     print(end="\n\n")
#
# def label_encoder(df: pd.DataFrame, columns: list) -> pd.DataFrame:
#     result = df.copy()
#     le = LabelEncoder()
#     for col in columns:
#         le.fit(df[col])
#         label_encoded = le.transform(df[col])
#         result[col] = label_encoded
#         # display_info(le.classes_)
#     return result
#
# def one_hot_encoder(df: pd.DataFrame, columns: list) -> pd.DataFrame:
#     result = df.copy()
#     ohe = OneHotEncoder()
#
#     # 모든 범주형 변수를 하나의 데이터프레임으로 결합
#     combined_df = pd.DataFrame()
#
#     for col in columns:
#         combined_df[col] = df[col]
#         one_hot_encoded = ohe.fit_transform(combined_df) # OneHotEncoder를 적용하여 모든 범주형 변수를 인코딩
#         ohe_df = pd.DataFrame(one_hot_encoded.toarray(), columns=ohe.get_feature_names_out(combined_df.columns)) # 인코딩 결과를 데이터프레임으로 변환
#         result = pd.concat([result, ohe_df], axis=1) # 결과 데이터프레임과 원본 데이터프레임 결합
#
#         result.drop(col, axis=1, inplace=True)
#     return result
#
# def ordinal_encoder(df: pd.DataFrame, columns: list) -> pd.DataFrame:
#     result = df.copy()
#     encoder = OrdinalEncoder()
#
#     # 범주형 변수에 대해 Ordinal Encoding 수행
#     for col in columns:
#         data_encoded = encoder.fit_transform(df[[col]])
#         result[col] = data_encoded
#
#     return result
#
# def standard_scaler(df: pd.DataFrame, columns: list) -> pd.DataFrame:
#     result = df.copy()
#     # result.drop('id', axis=1, inplace=True)
#     target_df = result.loc[:, columns]
#
#     scaler = StandardScaler()
#
#     scaler.fit(target_df)
#     scaled_data = scaler.transform(target_df)
#     scaled_data_df = pd.DataFrame(scaled_data, columns=columns)
#     scaled_data_df['id'] = result['id']
#     # pd.concat([result['id'], scaled_data_df], axis=1)
#     # scaled_data_df.reset_index()
#     # result.reset_index(d)
#     for col in columns:
#         result[col] = scaled_data_df[col]
#
#     # result = pd.concat([result, scaled_data_df])
#     # result = pd.concat([result], axis=1)
#     return result
#
# def numeric_transformer(df: pd.DataFrame, columns: list, strategy: str = 'B', viz_available: bool = False, viz_bins: int = 40) -> pd.DataFrame:
#     result = df.copy()
#     result_id = result['id']
#     # result.drop('id', axis=1, inplace=True)
#
#     try:
#         if strategy == 'L': # Log
#             for col in columns:
#                 result[col] = np.log1p(result[col])
#                 if viz_available:
#                     result[col].hist(bins=viz_bins)
#
#         if strategy == 'B': # Box-Cox
#             trans = PowerTransformer(method='box-cox')
#             for col in columns:
#                 result[col] = trans.fit_transform(result[col].values.reshape(-1, 1) + 1)
#                 if viz_available:
#                     result[col].hist(bins=viz_bins)
#     except ValueError:
#         print("ValueError: The Box-Cox transformation can only be applied to strictly positive data.")
#         print("System: Automatically change strategy as 'Quantile'")
#         strategy = 'Q'
#
#     if strategy == 'Y': # Yeo-Johnson
#         trans = PowerTransformer(method='yeo-johnson')
#         for col in columns:
#             result[col] = trans.fit_transform(result[col].values.reshape(-1, 1))
#             if viz_available:
#                 result[col].hist(bins=viz_bins)
#
#     elif strategy == 'Q': # Quantile
#         trans = QuantileTransformer(output_distribution='normal')
#         for col in columns:
#             result[col] = trans.fit_transform(result[col].values.reshape(-1, 1))
#             if viz_available:
#                 result[col].hist(bins=viz_bins)
#
#     else:
#         print('System: Please appropriate strategy.')
#
#     result['id'] = result_id
#     # result = pd.concat([df['id'], result], axis=1)
#     return result
#
# def control_imbalance(train_data: pd.DataFrame, label: pd.Series, seed: int = 42):
#     smote = SMOTE(random_state=seed)
#     resampled_train, resampled_label = smote.fit_resample(train_data, label)
#     resampled_train = pd.DataFrame(resampled_train, columns=train_data.columns)
#     resampled_label = pd.Series(resampled_label, name=label.name)
#     return resampled_train, resampled_label
#
# def run_encoding(df: pd.DataFrame, drop_col: list) -> pd.DataFrame:
#     encoded_labelling_data = df.copy()
#     encoded_labelling_data.reset_index(inplace=True)
#     encoded_labelling_data.drop(drop_col, axis=1, inplace=True)
#
#     encoded_labelling_data = label_encoder(df=encoded_labelling_data,
#                                            columns=['workclass', 'marital_status', 'occupation', 'relationship',
#                                                     'race',
#                                                     'native_country'])
#     encoded_onehot_data = one_hot_encoder(df=encoded_labelling_data, columns=['sex'])
#     encoded_ordinal_data = ordinal_encoder(df=encoded_onehot_data, columns=['education'])
#     # encoded_data = pd.concat([df['id'], encoded_ordinal_data], axis=1)
#
#     return encoded_ordinal_data
#
# def run(train_data: pd.DataFrame, test: pd.DataFrame, label: pd.Series, target_col: str, verbose: int = 0):
#     # yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
#     # target_col = 'target'
#
#     # set_initial_setting()
#
#     # # 데이터 읽기
#     # train_data, test, label = DataReader.read_from_db(
#     #     yaml_file_path=yaml_file_path,
#     #     table_name="adult_income",
#     #     label_col_name=target_col)
#
#     # 컬럼명의 '.' 문자 변경
#     train_data.columns = train_data.columns.str.replace('.', '_')
#     test.columns = test.columns.str.replace('.', '_')
#
#     # 범주형, 수치형 변수 분리
#     cat_columns = train_data.select_dtypes(include='object').columns
#     num_columns = train_data.select_dtypes(exclude='object').columns
#
#     # train에만 존재하는 샘플값 삭제
#     train_data.drop(index=8335, axis =0, inplace=True)
#
#     if verbose == 2:
#         eda_result_print(train_data, test, label, num_columns, cat_columns)
#
#     # 고려할 컬럼
#     # missing_col = ['workclass', 'occupation', 'native_country', 'race']
#     missing_cols = train_data.columns[train_data.isna().any()].tolist()
#     column_combinations = list(itertools.combinations(missing_cols, 2))
#     # column_combinations = list(itertools.combinations(cat_columns, 2))
#     if verbose == 1:
#         display_info(obj=column_combinations, msg="column_combinations")
#
#     if verbose == 1:
#         for combination in column_combinations:
#             cross_tab = pd.crosstab(train_data[combination[0]], train_data[combination[1]])
#             chi2, p, _, _ = chi2_contingency(cross_tab)
#             # display_info(cross_tab, f"Cross tabulation for columns {combination}")
#             display_info(obj=chi2, msg=f"chi2 for columns {combination}")
#             display_info(obj=p, msg=f"P-value for columns {combination}")
#
#             # # 시각화
#             if verbose == 3:
#                 sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g')
#                 plt.title('Cross Tabulation Heatmap')
#                 plt.show()
#
#     # 결측치 처리: 결측치는 Train 데이터에만 존재함
#     # native_country 결측치 처리
#     # tmp_train_data = train_data.copy()
#     # tmp_train_data.loc[(tmp_train_data['race'] == 'White') & (tmp_train_data['native_country'].isin([np.nan, None, 'None'])), 'native_country'] = 'United-States'
#     # tmp_train_data.loc[tmp_train_data['race'] != 'White', 'native_country'] = 'Others'
#     #
#     # # workclass 결측치 처리
#     # tmp_train_data['workclass'] = tmp_train_data['workclass'].fillna('Never-worked')
#     # tmp_train_data.loc[tmp_train_data['workclass'].isin([np.nan, None, 'None']), 'workclass'] = 'Never-worked'
#     #
#     # # occupation 결측치 처리
#     # tmp_train_data['occupation'] = tmp_train_data['occupation'].fillna('Others')
#     # tmp_train_data.loc[tmp_train_data['occupation'].isin([np.nan, None, 'None']), 'occupation'] = 'Others'
#
#     train_data.drop('index', axis=1, inplace=True)
#     train_data.reset_index(inplace=True)
#     tmp_train_data = pd.concat([train_data, label], axis=1)
#     tmp_train_data = tmp_train_data.dropna()
#     label = tmp_train_data['target']
#
#
#
#     # 결과 확인
#     if verbose == 1:
#         display_info(obj=pd.isna(train_data).sum(), msg="결측치 처리 결과")
#
#     train_data = tmp_train_data
#
#     drop_col = 'fnlwgt'
#     try:
#         train_data.drop(drop_col, axis=1, inplace=True)
#         test.drop(drop_col, axis=1, inplace=True)
#         num_columns = num_columns.drop(drop_col)
#     except KeyError:
#         print(f'KeyError: [{drop_col}] not found in axis"')
#
#     if verbose == 1:
#         eda_result_print(
#             train_data=train_data,
#             test=test,
#             label=label,
#             num_columns=num_columns,
#             cat_columns=cat_columns,
#             target_col=target_col
#         )
#     # 레이블이 0인 데이터가 레이블이 1인 데이터보다 약 3배 정도 많다.
#     '''
#     imbalanced 처리 방법
#
#     SMOTE(Synthetic Minority Over-sampling Technique)
#     클래스 가중치 조정
#     AdaBoost나 Gradient Boosting 등의 앙상블 기법
#     정밀도(precision), 재현율(recall), F1 점수 등의 평가 지표를 사용
#
#     여러 기법을 조합하여 사용하는 것이 효과적
#     '''
#
#     # ENCODING
#     if verbose == 1:
#         print(' ##### Encdoing ##### ')
#     # categorical column들 중 어느 컬럼에 어느 방법을 적용해야 하는가?
#     '''Index(['workclass', 'education', 'marital_status', 'occupation',
#        'relationship', 'race', 'sex', 'native_country'],
#       dtype='object')'''
#     '''
#     - Label: 단순 인코딩; 0, 1, 2: workclass, marital_status, occupation, relationship, race, native_country
#     - One-Hot: 0, 1 값만을 가진 새로운 차원 생성: sex
#     , Ordinal: 순서가 중요한 경우: education
#     '''
#
#     drop_col = ['index', 'level_0']
#     train_data = run_encoding(train_data, drop_col)
#     test = run_encoding(test, drop_col)
#
#     for col in drop_col:
#         if col in num_columns:
#             num_columns = num_columns.drop(col)
#
#     train_data = standard_scaler(df=train_data, columns=num_columns)
#     test = standard_scaler(df=test, columns=num_columns)
#
#     train_data = numeric_transformer(df=train_data, columns=num_columns, strategy='Q', viz_available=True)
#     test = numeric_transformer(df=test, columns=num_columns, strategy='Q', viz_available=True)
#
#     train_data, label = control_imbalance(train_data=train_data, label=label)
#
#     return train_data, test, label


import os
import warnings
import itertools
import pprint
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer, \
    QuantileTransformer


def set_initial_setting():
    warnings.filterwarnings(action='ignore')
    plt.style.use("ggplot")
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.max_columns', 500)

    # Seed 설정
    seed_num = 42
    np.random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)


def display_info(obj=None, msg=""):
    pprint.pp(f" ##### {msg} ##### ")
    pprint.pp(obj)
    print()


def eda_result_print(train_data, test, label, num_columns, cat_columns, target_col):
    display_info(train_data, "Train Data")
    display_info(test, "Test Data")

    display_info(train_data, "Train Data loaded")
    display_info(train_data.head(), "Head of Train Data")
    display_info(train_data.shape, "Shape of Train Data")
    display_info(test.shape, "Shape of Test Data")
    display_info(train_data.info(), "Info of Train Data")
    display_info(test.info(), "Info of Test Data")
    display_info(label.info(), "Info of Label")
    display_info(train_data.describe(include='all'), "Description of Train Data")
    display_info(test.describe(include='all'), "Description of Test Data")

    for col in num_columns:
        display_info(train_data[col], col)
        display_info(train_data[col].describe(), f"{col}'s Description")
        display_info(train_data[col].unique(), f"{col}'s Unique Values")

    for col in cat_columns:
        display_info(train_data[col], col)
        tmp_df = train_data[col].describe()
        display_info(tmp_df, f"{col}'s Description")
        ratios = train_data[col].value_counts() / len(train_data[col])
        display_info(ratios, f"Ratio of {col}.")
        display_info(test[col].unique(), f"{col}'s Unique Values")

    # 결측치 확인
    display_info(cat_columns, "Cat Columns")
    display_info(num_columns, "Num Columns")
    display_info(train_data.isna().sum(), "NaNs in Train Data")
    display_info(test.isna().sum(), "NaNs in Test")

    # 클래스 imbalanced 확인
    for idx, data in enumerate([train_data]):
        if idx == 0:
            print(f"Target:{target_col} imbalanced 확인")

        tmp_data = pd.concat([data, label], axis=1)

        total_count = len(tmp_data)
        unique_values = tmp_data[target_col].unique()
        for value in unique_values:
            count = (tmp_data[target_col] == value).sum()
            ratio = count / total_count
            print(f"Value: {value}, Count: {count}, Ratio: {ratio:.4f}")
        print()


def label_encoder(df, columns):
    result = df.copy()
    le = LabelEncoder()
    for col in columns:
        le.fit(df[col])
        label_encoded = le.transform(df[col])
        result[col] = label_encoded
    return result


def one_hot_encoder(df, columns):
    result = df.copy()
    ohe = OneHotEncoder()

    combined_df = pd.DataFrame()

    for col in columns:
        combined_df[col] = df[col]
        one_hot_encoded = ohe.fit_transform(combined_df)
        ohe_df = pd.DataFrame(one_hot_encoded.toarray(), columns=ohe.get_feature_names_out(combined_df.columns))
        result = pd.concat([result, ohe_df], axis=1)

        result.drop(col, axis=1, inplace=True)
    return result


def ordinal_encoder(df, columns):
    result = df.copy()
    encoder = OrdinalEncoder()

    for col in columns:
        data_encoded = encoder.fit_transform(df[[col]])
        result[col] = data_encoded

    return result


def standard_scaler(df, columns):
    result = df.copy()
    target_df = result.loc[:, columns]

    scaler = StandardScaler()
    scaler.fit(target_df)
    scaled_data = scaler.transform(target_df)
    scaled_data_df = pd.DataFrame(scaled_data, columns=columns)

    for col in columns:
        result[col] = scaled_data_df[col]

    return result


def numeric_transformer(df, columns, strategy='Q', viz_available=False, viz_bins=40):
    result = df.copy()

    try:
        if strategy == 'L':  # Log
            for col in columns:
                result[col] = np.log1p(result[col])

        if strategy == 'B':  # Box-Cox
            trans = PowerTransformer(method='box-cox')
            for col in columns:
                result[col] = trans.fit_transform(result[col].values.reshape(-1, 1) + 1)

    except ValueError:
        print("ValueError: The Box-Cox transformation can only be applied to strictly positive data.")
        print("System: Automatically change strategy as 'Quantile'")
        strategy = 'Q'

    if strategy == 'Y':  # Yeo-Johnson
        trans = PowerTransformer(method='yeo-johnson')
        for col in columns:
            result[col] = trans.fit_transform(result[col].values.reshape(-1, 1))

    elif strategy == 'Q':  # Quantile
        trans = QuantileTransformer(output_distribution='normal')
        for col in columns:
            result[col] = trans.fit_transform(result[col].values.reshape(-1, 1))

    else:
        print('System: Please appropriate strategy.')

    return result


def control_imbalance(train_data, label, seed=42):
    smote = SMOTE(random_state=seed)
    resampled_train, resampled_label = smote.fit_resample(train_data, label)
    resampled_train = pd.DataFrame(resampled_train, columns=train_data.columns)
    resampled_label = pd.Series(resampled_label, name=label.name)
    return resampled_train, resampled_label


def run_encoding(df, drop_col):
    encoded_labelling_data = df.copy()
    encoded_labelling_data.reset_index(inplace=True)
    encoded_labelling_data.drop(drop_col, axis=1, inplace=True)

    encoded_labelling_data = label_encoder(df=encoded_labelling_data,
                                           columns=['workclass', 'marital_status', 'occupation', 'relationship',
                                                    'race',
                                                    'native_country'])
    encoded_onehot_data = one_hot_encoder(df=encoded_labelling_data, columns=['sex'])
    encoded_ordinal_data = ordinal_encoder(df=encoded_onehot_data, columns=['education'])

    return encoded_ordinal_data


def run(train_data, test, label, target_col, verbose=0):
    train_data.columns = train_data.columns.str.replace('.', '_')
    test.columns = test.columns.str.replace('.', '_')

    cat_columns = train_data.select_dtypes(include='object').columns
    num_columns = train_data.select_dtypes(exclude='object').columns

    train_data.drop(index=8335, axis=0, inplace=True)

    if verbose == 2:
        eda_result_print(train_data, test, label, num_columns, cat_columns)

    missing_cols = train_data.columns[train_data.isna().any()].tolist()
    column_combinations = list(itertools.combinations(missing_cols, 2))

    if verbose == 1:
        display_info(column_combinations, "column_combinations")

    if verbose == 1:
        for combination in column_combinations:
            cross_tab = pd.crosstab(train_data[combination[0]], train_data[combination[1]])
            chi2, p, _, _ = chi2_contingency(cross_tab)
            display_info(chi2, f"chi2 for columns {combination}")
            display_info(p, f"P-value for columns {combination}")

            if verbose == 3:
                sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g')
                plt.title('Cross Tabulation Heatmap')
                plt.show()

    train_data.drop('index', axis=1, inplace=True)
    train_data.reset_index(inplace=True)
    tmp_train_data = pd.concat([train_data, label], axis=1)
    tmp_train_data = tmp_train_data.dropna()
    label = tmp_train_data['target']

    train_data = tmp_train_data

    drop_col = 'fnlwgt'
    try:
        train_data.drop(drop_col, axis=1, inplace=True)
        test.drop(drop_col, axis=1, inplace=True)
        num_columns = num_columns.drop(drop_col)
    except KeyError:
        print(f'KeyError: [{drop_col}] not found in axis"')

    if verbose == 1:
        eda_result_print(
            train_data=train_data,
            test=test,
            label=label,
            num_columns=num_columns,
            cat_columns=cat_columns,
            target_col=target_col
        )

    # 결측치 처리: 결측치는 Train 데이터에만 존재함
    # native_country 결측치 처리
    # tmp_train_data = train_data.copy()
    # tmp_train_data.loc[(tmp_train_data['race'] == 'White') & (tmp_train_data['native_country'].isin([np.nan, None, 'None'])), 'native_country'] = 'United-States'
    # tmp_train_data.loc[tmp_train_data['race'] != 'White', 'native_country'] = 'Others'
    #
    # # workclass 결측치 처리
    # tmp_train_data['workclass'] = tmp_train_data['workclass'].fillna('Never-worked')
    # tmp_train_data.loc[tmp_train_data['workclass'].isin([np.nan, None, 'None']), 'workclass'] = 'Never-worked'
    #
    # # occupation 결측치 처리
    # tmp_train_data['occupation'] = tmp_train_data['occupation'].fillna('Others')
    # tmp_train_data.loc[tmp_train_data['occupation'].isin([np.nan, None, 'None']), 'occupation'] = 'Others'

    if verbose == 1:
        print(' ##### Encoding ##### ')

    drop_col = ['index', 'level_0']
    train_data = run_encoding(train_data, drop_col)
    test = run_encoding(test, drop_col)

    for col in drop_col:
        if col in num_columns:
            num_columns = num_columns.drop(col)

    train_data = standard_scaler(train_data, num_columns)
    test = standard_scaler(test, num_columns)

    train_data = numeric_transformer(train_data, num_columns, strategy='Q', viz_available=True)
    test = numeric_transformer(test, num_columns, strategy='Q', viz_available=True)

    train_data, label = control_imbalance(train_data, label)

    train_data.drop('target', axis=1, inplace=True)
    train_data.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)

    return train_data, test, label
