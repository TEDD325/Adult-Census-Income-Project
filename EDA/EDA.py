import os
import pprint
from os.path import join
from pathlib import Path
import copy
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

def display_info(obj: object, title: str = ""):
    pprint.pp(" ##### " + title + " ##### ")
    pprint.pp(obj)
    print(end="\n\n")

ROOT_PATH = os.getcwd()
train_data_path = join(ROOT_PATH, '../data/', 'train.csv')

train_data = pd.read_csv(train_data_path)

display_info(train_data, "Train Data loaded")

