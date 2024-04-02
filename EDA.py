import os
import warnings
import DataReader
from scipy.stats import chi2_contingency
import itertools
import pprint

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
# import pandas_profiling
import seaborn as sns
import random as rn
import os
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn import metrics

import xgboost as xgb
import lightgbm as lgb
import catboost as catboost

from collections import Counter
import warnings

import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import pandas as pd
import sys
from scipy.stats import chi2_contingency
from scipy.stats import kstest
from statsmodels.formula.api import ols

warnings.filterwarnings(action='ignore')
plt.style.use("ggplot")
pd.set_option('display.max_rows',30)
pd.set_option('display.max_columns',500)

yaml_file_path = os.path.dirname(os.path.realpath(__file__)) + '/info.yaml'
target_col = 'target'

def check_lib_version():
    print("numpy version: {}".format(np.__version__))
    print("pandas version: {}".format(pd.__version__))
    print("matplotlib version: {}".format(matplotlib.__version__))
    print("scikit-learn version: {}".format(sklearn.__version__))
    print("xgboost version: {}".format(xgb.__version__))
    print("lightgbm version: {}".format(lgb.__version__))
    print("catboost version: {}".format(catboost.__version__))

def display_info(obj: object, title: str = ""):
    print()
    pprint.pp(" ##### " + title + " ##### ")
    pprint.pp(obj)
    print(end="\n\n")

ROOT_PATH = os.getcwd()

train_data, test, label = DataReader.read_from_db(
    yaml_file_path=yaml_file_path,
    table_name="adult_income",
    label_col_name=target_col)

seed_num = 42   ####
np.random.seed(seed_num)
rn.seed(seed_num)
os.environ['PYTHONHASHSEED']=str(seed_num)

train_data.columns = train_data.columns.str.replace('.','_')
test.columns = test.columns.str.replace('.','_')

cat_columns = train_data.select_dtypes(include='object').columns # 범주형 변수
num_columns = train_data.select_dtypes(exclude='object').columns # 수치형 변수

# check_lib_version()
# display_info(train_data, "Train Data loaded")

# display_info(train_data.head(), "Head of Train Data")
# display_info(train_data.shape, "Shape of Train Data") # (17480, 17)
# display_info(test.shape, "Shape of Test Data") # (15081, 16)
# display_info(train_data.info(), "Info of Train Data")
# display_info(test.info(), "Info of Test Data")
# display_info(label.info(), "Info of Label")
# display_info(train_data.describe(include='all'), "Description of Train Data")
# display_info(test.describe(include='all'), "Description of Test Data")
#
# for col in num_columns:
#     display_info(train_data[col], col)
#     display_info(train_data[col].describe(), col+"'s Description")
#     display_info(train_data[col].unique(), col+"'s Unique Values")
#
# for col in cat_columns:
#     display_info(train_data[col], col)
#     display_info(test[col].describe(), col + "'s Description")
#     display_info(test[col].unique(), col + "'s Unique Values")

display_info(cat_columns, "Cat Columns")
display_info(num_columns, "Num Columns")
display_info(train_data.isna().sum(), "NaNs in Train Data")
display_info(test.isna().sum(), "NaNs in Test")

'''
# 변수 구분
범주형 변수: 'workclass', 'education', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'native_country'
수치형 변수: 'index', 'id', 'age', 'fnlwgt', 'education_num', 'capital_gain',
       'capital_loss', 'hours_per_week', 'target'

# shape
"Shape of Train Data": (17480, 16)
"Shape of Test Data": (15081, 16)
test 샘플이 많은 것 같다.

# info 결과 분석
- train_data에는 workclass, occupation이 거의 비슷한 개수로, native_country에서 null이 존재함
workclass         1836
occupation        1843
native_country     583
- test에는 null값이 존재하지 않음
- label은 모두 수치형 변수들이라 인코딩이 필요할 것 같음

# describe 결과 분석
각 컬럼 별로 min, max값이 다르다 -> 스케일링 필요

# 컬럼별 분석
index와 id가 동일하다.
age의 분포를 봐야할 것 같다.
nlwgt: CPS(Current Population Survey) 가중치: 중요할까? => 레퍼런스 결과 제외하는 편
education_num 최소값 최대값 추출이 필요해보임>
    count    17480.000000
    mean        10.036556
    std          2.604415
    min          1.000000
    25%          9.000000
    50%         10.000000
    75%         12.000000
    max         16.000000
    -> order가 중요해보이는데 맞을까? yes. 
    => age, education_num, hours_per_week

capital_gain은 Min값이 0, max값이 99999이며, 75% 퀀타일에도 0값이 들어가있을 정도로 편중된 데이터로 보인다. 분포를 살펴볼 필요가 있다.
capital_loss도 마찬가지.

sex는 남녀 성비가 안 맞는다.
인종은 백인에 편중된다.
workclass는 private에 편중된다.
native_country는 미국에 편중된다.
education은 1st-4th처럼 일부 샘플들이 그룹으로 묶인 상태다.

'''

'''
결측치 처리에 대한 인사이트를 얻기 위해 NaN값이 있는 컬럼끼리의 상관관계를 살펴보아야 할 것 같다.
workclass: 범주형
occupation: 범주형
native_country: 범주형


다음의 인사이트가 제안된다.
결측치 삭제: 가장 간단한 방법. 데이터의 양이 충분하고 결측치가 일부인 경우에만 적합 
    - 이 데이터에서 가장 많은 결측치를 보이는 workclass의 결측치 샘플 개수는 전체의 10% 정도다. 
    - 단, 결측치의 비율이 작더라도 해당 데이터가 분석에 중요한 역할을 하는지, 결측치의 원인이 무엇인지 등을 고려하여 적절한 결정을 내릴 필요가 있다.
    - 10% 정도의 결측치를 "일부"로 간주할 수 있을까?
    - 다른 변수들에 대한 결측치 유무와 데이터 분포 등을 고려하여 종합적으로 판단하는 것이 좋다.
대체값 사용: 결측치를 대체할 수 있는 다른 값으로 채운다. 이를 위해 평균, 중앙값, 최빈값 등의 대체값을 사용할 수 있다. 단, 데이터의 분포를 왜곡할 수 있다.
    - Mode 방식으로 범주형 결측치를 채우기에는 특정 값에 너무 편중될 것 같다.
새로운 범주 추가: 결측치를 새로운 범주로 취급한다. 이 방법은 결측치가 다른 의미를 갖는 경우에 유용하다.
    - 
예측 모델 사용: 다른 변수를 기반으로 결측치를 예측하는 모델을 사용하여 결측치를 대체한다. 예를 들어, 회귀 모델이나 분류 모델을 사용하여 결측치를 예측할 수 있다. -> 또 다른 모델링을 진행해야 하는데, 프로젝트 기간 상 부담스럽다.
다중 대체값 사용: 여러 가지 대체값을 시도하고 각각의 결과를 비교하여 최적의 대체값을 선택한다. 이 방법은 다양한 상황에 대응할 수 있다. -> 현실적으로 너무 오래 걸릴 것으로 보인다.
'''

# 결측치를 보이는 컬럼
missing_col = ['workclass', 'occupation', 'native_country']
column_combinations = list(itertools.combinations(missing_col, 2))
display_info(column_combinations, "column_combinations")

# 결측치들 간에 상관관계 분석
# 크로스탭 값이 높을 때, 카이제곱값이 높을 때, P 값이 낮을 때 두 범주형 변수 간의 상관관계가 높다.
'''
크로스탭 값이 높을 때: 크로스탭의 값이 높다는 것은 해당 조합에 대해 해당하는 데이터가 많다는 것을 의미한다. 이는 두 변수 간의 관계가 강하다는 신호일 수도 있다. 카이제곱 등을 추가적으로 살펴봐야 한다.
카이제곱값이 높을 때: 카이제곱값은 카이제곱 검정을 통해 계산되며, 두 범주형 변수 간의 독립성 여부를 판단하는 데 사용된다. 따라서 카이제곱값이 높을수록 두 변수 간의 관계가 강하고 독립적이지 않다는 것을 의미한다.
P 값이 낮을 때: P 값은 카이제곱 검정의 유의확률로, 관측된 데이터가 귀무가설에 부합하는지 여부를 나타낸다. P 값이 낮을수록 귀무가설을 기각하고 대립가설을 채택할 확률이 높으며, 이는 두 변수 간의 관계가 통계적으로 유의미하다는 것을 의미한다.
카이제곱값과 p값 중에서는, P값을 더 유의한다.
만약 카이제곱값이 높지만 P 값이 높다면, 이는 관측된 관련성이 우연에 의한 것일 가능성이 높다는 것을 의미한다.
'''
# for combination in column_combinations:
#     cross_tab = pd.crosstab(train_data[combination[0]], train_data[combination[1]])
#     chi2, p, _, _ = chi2_contingency(cross_tab)
#     display_info(cross_tab, f"Cross tabulation for columns {combination}")
#     display_info(chi2, f"chi2 for columns {combination}")
#     display_info(p, f"P-value for columns {combination}")

    # sns.heatmap(cross_tab, annot=True, cmap='coolwarm', fmt='g')
    # plt.title('Cross Tabulation Heatmap')
    # plt.show()

'''
'workclass', 'occupation'은 우연에 의한 관련성
'workclass', 'native_country' 보다는 'occupation', 'native_country'의 상관성이 더 높다.
'''


type1= train_data['occupation'].isna()
type2= train_data['native_country'].isna()
two_na=train_data[type1&type2]['occupation', 'native_country']
display_info(two_na, "NaN of occupation & native_country")
display_info(two_na, "Shape of two_na") # 1836개



print("Debugging point")