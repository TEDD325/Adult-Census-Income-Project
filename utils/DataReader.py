import multiprocessing
import yaml
import pandas as pd
from sqlalchemy import create_engine, text

n_cpus = multiprocessing.cpu_count() - 1

def read_yaml(file_path: str):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def read_from_db(yaml_file_path: str, table_name: str, label_col_name: str):
    global_info = read_yaml(yaml_file_path)

    try:
        engine = create_engine(url='postgresql+psycopg2://'+global_info['db']['id']+':'+global_info['db']['pw']+'@'+global_info['db']['host']+':'+global_info['db']['port']+'/'+global_info['db']['table'])
        conn = engine.connect()

        sql_for_data = 'SELECT * FROM '+table_name+'_train'
        sql_for_test = 'SELECT * from '+table_name+'_test'

        data = pd.read_sql(text(sql_for_data), con=conn)
        test = pd.read_sql(text(sql_for_test), con=conn)
        #
        label = data[label_col_name]

        # data = pd.read_sql('SELECT * FROM adult_income_train', con=engine)
        # test = pd.read_sql('SELECT * from adult_income_test', con=engine)

        # data = pd.read_sql(text('SELECT * FROM adult_income_train'), con=conn)
        #
        # label = data[label_col_name]
        #
        # test = pd.read_sql(text('SELECT * from adult_income_test'), con=conn)


    except:
        print('Occurred Error caused by Reading from db')

    return data, test, label