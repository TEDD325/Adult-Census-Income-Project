import pandas as pd
from sqlalchemy import create_engine, text
from Utils import Utils


def read_from_db(yaml_file_path: str, table_name: str, label_col_name: str):
    try:
        global_info = Utils.read_yaml(yaml_file_path)

        engine = create_engine(url='postgresql+psycopg2://'+global_info['db']['id']+':'+global_info['db']['pw']+'@'+global_info['db']['host']+':'+global_info['db']['port']+'/'+global_info['db']['table'])

        # 데이터베이스 연결
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