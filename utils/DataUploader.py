import pandas as pd
from sqlalchemy import create_engine
from utils.Utils import Utils


def upload_to_db(yaml_file_path: str, data_path: str, table_name: str):
    global_info = Utils.read_yaml()(yaml_file_path)

    try:
        engine = create_engine(url='postgresql+psycopg2://'+global_info['db']['id']+':'+global_info['db']['pw']+'@'+global_info['db']['host']+':'+global_info['db']['port']+'/'+global_info['db']['table'])
        train_df = pd.read_csv(data_path+'train.csv')
        test_df = pd.read_csv(data_path+'test.csv')

        train_df.to_sql(table_name + '_train', engine, if_exists='append')
        test_df.to_sql(table_name + '_test', engine, if_exists='append')
    except:
        print('Occurred Error caused by DB Uploading')