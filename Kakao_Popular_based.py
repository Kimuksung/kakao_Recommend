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

def chainer(s):
    return list(chain.from_iterable(s.str.split(' ')))

read_cnt_by_user = read['article_id'].str.split(' ').map(len)

read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                         'hr': np.repeat(read['hr'], read_cnt_by_user),
                         'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                         'article_id': chainer(read['article_id'])})

metadata.info()
metadata.head()

read_raw.info()
read_raw.head()

popular_read_dict = {}
popular = read_raw[read_raw.dt.between('20190215' , '20190228')]
for i in popular['article_id']:
    if i in popular_read_dict.keys():
       popular_read_dict[i] += 1
    else:
        popular_read_dict[i] = 1

popular_len = int(len(popular_read_dict) * 0.2)

popular_read = sorted(popular_read_dict.items() , key = (lambda x:x[1]) , reverse =True)[:popular_len]


tmp = []
from datetime import datetime
for i in range(len(metadata.reg_ts)):
    ts = int(metadata.reg_ts[i]/1000)
    tmp.append(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
    print(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))

metadata['date'] = tmp
metadata.head()



