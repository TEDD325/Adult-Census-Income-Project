from utils.DataUploader import *

yaml_file_path = './info.yaml'
data_path = './data/'
upload_to_db(yaml_file_path=yaml_file_path, data_path=data_path, table_name="Adult Census Income")
