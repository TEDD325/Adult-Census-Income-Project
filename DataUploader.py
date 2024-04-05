from sqlalchemy import create_engine
import pandas as pd
import yaml
import os

def upload_to_db(yaml_file_path: str, table_name: str, data_path: str = None, train_df: pd.DataFrame = None,
                 test_df: pd.DataFrame = None):
    try:
        # YAML 파일에서 데이터베이스 정보 읽기
        with open(yaml_file_path, 'r') as file:
            global_info = yaml.safe_load(file)

        # 데이터 경로가 제공되지 않은 경우 예외 발생
        if not data_path and not train_df and not test_df:
            raise ValueError("Please provide the parameter data_path, train_df, or test_df")

        # 데이터베이스 연결 엔진 생성
        engine = create_engine(url='postgresql+psycopg2://'+global_info['db']['id']+':'+global_info['db']['pw']+'@'+global_info['db']['host']+':'+global_info['db']['port']+'/'+global_info['db']['table'])

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