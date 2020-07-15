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
pd.set_option('display.max_rows', 50)

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

read_raw.info()
read_raw.head()

# metadata 2018.10 ~ 2019.3.14
# data preprocessing unix data -> real data
def trans_unix(metadata):
    tmp = []
    from datetime import datetime
    for i in range(len(metadata.reg_ts)):
        ts = int(metadata.reg_ts[i]/1000)
        tmp.append(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
        print(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
    
    metadata['date'] = tmp
    metadata.head()
    metadata = metadata.drop('reg_ts' , axis=1)
    return metadata

metadata = trans_unix(metadata)
#metadata 원하는 기간 데이터만 추출
metadata = metadata[metadata.date.between('20181001' , '20190314')]

# recent1 대해 수행 중
# metadata를 통해 2019.02.15~02.28
def make_recent1_metadata(metadata):
    recent_metadata = metadata[metadata.date.between('20190215' , '20190228')]
    recent_metadata.id
    
    read_raw # 22110706
    recent_read_popular = read_raw[read_raw.article_id.isin(recent_metadata.id)] # 930498
    recent_read_popular.article_id
    
    popular_read_dict ={}
    for i in recent_read_popular['article_id']:
        if i in popular_read_dict.keys():
           popular_read_dict[i] += 1
        else:
            popular_read_dict[i] = 1
    
    len(popular_read_dict) # 10368
    popular_len = int(len(popular_read_dict) * 0.2) # 2073
    popular_read = sorted(popular_read_dict.items() , key = (lambda x:x[1]) , reverse =True)[:popular_len]
    tmp = []
    for i,j in popular_read:
        tmp.append(i)

    return metadata[metadata.id.isin(tmp)]

recent1_popular = make_recent1_metadata(metadata)

# 임의의 random data로 test
# user #ae0aa864580107263a99c5087eb27a9d
# following based
tmp = metadata[metadata.date.between('20190222' , '20190228')].sample(n=1)

#test 
# 전체일자에 대해서는 815개를 소비 -> 해당 기간안에서는 128개 data 소비
# 구독 작가 : 53명
# 구독 작가 중 최근에 읽은 작가 : 1명(18번)
# 이 경우에는 기간 안에는 존재하지만 인기글이 아니여서 추천되지 않는다.

read_raw[read_raw['article_id'] == "@dogflower_17"]
temp = read_raw[read_raw.user_id=="#ae0aa864580107263a99c5087eb27a9d"] # 815
compare1= temp[temp.dt.between('20190222','20190228')].article_id #128
compare1

# 기간 내에는 128개의 read를 소비

users[users.id=="#ae0aa864580107263a99c5087eb27a9d"]
tmp2 = users[users.id=="#ae0aa864580107263a99c5087eb27a9d"].following_list
def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt
tmp2 = tmp2.tolist()
tmp2 = flatten(tmp2)
len(tmp2) # 구독한 작가는 53명

following_dict = {}
for i in tmp2:
    if not i in following_dict.keys(): 
        following_dict[i] = 0    

for i in compare1:
    text = re.sub('[\_\d]', '', i)
    print(text)
    if text in following_dict.keys():
        following_dict[text] +=1
 
following_dict = {key:val for key, val in following_dict.items() if val != 0}
list(following_dict.keys())

metadata[metadata.user_id.isin(list(following_dict.keys()))]    
recent1_popular[recent1_popular.user_id.isin(list(following_dict.keys()))]

