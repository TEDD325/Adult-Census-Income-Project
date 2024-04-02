from DataReader import *
import pprint

# global variable
yaml_file_path = './info.yaml'
data_path = './data/'
target_col = 'target'

# ##### uploading data to PoestgreDB
# upload_to_db(yaml_file_path=yaml_file_path, data_path=data_path, table_name="adult_income")

# ##### Read data from PoestgreDB
data, test, label = read_from_db(
    yaml_file_path=yaml_file_path,
    table_name="adult_income",
    label_col_name=target_col)

pprint.pp("data: {}".format(data))
pprint.pp("test: {}".format(test))
pprint.pp("label: {}".format(label))