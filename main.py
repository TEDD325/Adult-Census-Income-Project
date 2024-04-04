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
import multiprocessing

n_cpus = multiprocessing.cpu_count()
n_jobs = max(1, n_cpus - 1)

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
        # 'iterations': trial.suggest_int('iterations', 300, 1000),
        'iterations': 937, # 386
        # 'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'learning_rate': 0.066, # 0.095
        # 'depth': trial.suggest_int('depth', 2, 10),
        'depth': 9, # 10
        'random_state': 42,
        # 'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 0.1, 10.0),
        # 'random_strength': trial.suggest_uniform('random_strength', 0.0, 1.0),
        # 'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
        # 'border_count': trial.suggest_int('border_count', 50, 255),
        # 'leaf_estimation_method': trial.suggest_categorical('leaf_estimation_method', ['Newton', 'Gradient']),
        'verbose': 100
    }
    # [I 2024-04-03 11:17:43,802] Trial 6 finished with value: 0.8980750499191027 and parameters: {'iterations': 937, 'learning_rate': 0.0655074888900418, 'depth': 9}. Best is trial 6 with value: 0.8980750499191027.
    # Trial 21 finished with value: 0.8859549295797973 and parameters: {'iterations': 368, 'learning_rate': 0.09502626806080669, 'depth': 10}. Best is trial 6 with value: 0.8980750499191027.

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize array to store average validation accuracys
    avg_val_accuracy = []

    # Perform cross validation
    for train_index, test_index in skf.split(train_data, label):
        x_train, x_valid = train_data.iloc[train_index], train_data.iloc[test_index]
        y_train, y_valid = label[train_index], label[test_index]

        # Train model with current hyperparameters
        model = CatBoostClassifier(**params, thread_count=n_jobs)
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
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=5)

    # Get best hyperparameters
    # best_params = study.best_params
    best_params = {
        'learning_rate': 0.05569734266093502,
         'l2_leaf_reg': 2.6630610410630625e-05,
         'od_type': 'IncToDec',
         'random_strength': 36,
         'boosting_type': 'Ordered',
         'bootstrap_type': 'MVS',
         'max_depth': 15,
         'n_estimators': 7651,
         'colsample_bylevel': 0.04351494898046096,
         'objective': 'Logloss',
         'max_bin': 322,
         'min_child_samples': 96}

    # Train final model with best hyperparameters
    model = CatBoostClassifier(**best_params, thread_count=n_jobs)
    model.fit(train_data, label, verbose=True)

    # Save Best Parameters
    # ** best_params

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