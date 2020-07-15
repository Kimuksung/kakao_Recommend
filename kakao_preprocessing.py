from collections import Counter
from datetime import timedelta, datetime
import glob
from itertools import chain
import json
import os
import re

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
pd.set_option('display.max_columns', 500)

# 1.Data Load
magazine = pd.read_json('magazine.json', lines=True)
magazine.shape
#(27967, 2)

magazine.head()

metadata = pd.read_json('metadata.json', lines=True)
metadata.shape

metadata.head()

users = pd.read_json('users.json', lines=True)
users.shape
users.head()
users.info()

read_file_lst = glob.glob('read/*')
exclude_file_lst = ['read.tar']

read_df_lst = []
for f in read_file_lst:
    file_name = os.path.basename(f)
    if file_name in exclude_file_lst:
        print(file_name)
    else:
        df_temp = pd.read_csv(f, header=None, names=['raw'])
        df_temp['dt'] = file_name[:8]
        df_temp['hr'] = file_name[8:10]
        df_temp['user_id'] = df_temp['raw'].str.split(' ').str[0]
        df_temp['article_id'] = df_temp['raw'].str.split(' ').str[1:].str.join(' ').str.strip()
        read_df_lst.append(df_temp)

read = pd.concat(read_df_lst)
read.shape
read.head()

# 2. Data 전처리
# 하나의 뭉쳐있는 Data를 여러 개의 Data로 분할
def chainer(s):
    return list(chain.from_iterable(s.str.split(' ')))

read_cnt_by_user = read['article_id'].str.split(' ').map(len)

read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                         'hr': np.repeat(read['hr'], read_cnt_by_user),
                         'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                         'article_id': chainer(read['article_id'])})

read_raw.shape

read_raw.head()

read_raw[['user_id', 'article_id']].drop_duplicates()


user_read = pd.merge(users , read_raw[['user_id', 'article_id']].drop_duplicates() , left_on='id' , right_on='user_id')
user_read = user_read.drop('id' , axis=1)
user_read.info()
user_read.shape
user_read.head()

df = pd.merge( user_read , metadata[['id' , 'title', 'sub_title' , 'keyword_list']] , left_on = 'article_id' , right_on = 'id')
df = df.drop('id' , axis = 1)
df.info()

df.head()

