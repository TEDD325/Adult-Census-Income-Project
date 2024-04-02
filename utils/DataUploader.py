import pandas as pd
import yaml
from sqlalchemy import create_engine, text
import multiprocessing

n_cpus = multiprocessing.cpu_count() - 1

def read_yaml(file_path: str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def upload_to_db(yaml_file_path: str, data_path: str, table_name: str):
    global_info = read_yaml(yaml_file_path)

    engine = create_engine(url='postgresql+psycopg2://'+global_info['db']['id']+':'+global_info['db']['pw']+'@'+global_info['db']['host']+':'+global_info['db']['port']+'/'+global_info['db']['table'])
    # conn = engine.connect()
    try:
        train_df = pd.read_csv(data_path+'train.csv')
        test_df = pd.read_csv(data_path+'test.csv')

        train_df.to_sql(table_name + ' Train', engine, if_exists='append')
        test_df.to_sql(table_name + ' Test', engine, if_exists='append')
    except:
        print('Occurred DB Uploading')