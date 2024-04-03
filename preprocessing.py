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
