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

    return metadata[metadata.id.isin(tmp)] , popular_read

recent1_popular , popular_read = make_recent1_metadata(metadata)

#아이디어는 2개
# 1. magazine에 대해서 user가 읽은 거에 대해 magazine별로 비율을 나누어 추천(인기도 높고 최신 순으로 추천할듯)
# 2. magazine에 대해 28000개 labeling에 대해 embedding 시키고 이를 이용하여 
# user가 magazine어떤 분야에 흥미가 있는지를 나타낼수있다.  

# 아이디어 1.
recent1_popular.sample(n=1)
# =============================================================================
#         magazine_id     user_id title      keyword_list  \
# 496111        34882  @bearhyang  운전빌런  [그림일기, 운전, 신혼부부]   
# 
#                                display_url sub_title  article_id  \
# 496111  https://brunch.co.kr/@bearhyang/42                    42   
# 
#                    id      date  
# 496111  @bearhyang_42  20190220  
# =============================================================================
# userid = #0c04b1be93b3c76750c46161c7990bfb


user_id = "#0c04b1be93b3c76750c46161c7990bfb"
user_article = read_raw[read_raw.user_id==user_id].article_id
user_article = list(set(user_article))

magazine_dict ={}
user_magazine = recent1_popular[recent1_popular.id.isin(user_article)].magazine_id
for i in user_magazine:
    if i in magazine_dict.keys():
        magazine_dict[i] +=1
    else:
        magazine_dict[i] = 1

def divide(dividends, num):
    ret = dict()
    for key, dividend in dividends.items():
        ret[key] = dividend/num
    return ret

# 인기도 순으로 나타내지 magazine의 id를 보고 상위 몇개만 추천
magazine_dict = sorted(magazine_dict.items(), key = (lambda x:x[1]) , reverse =True)
recent1_popular[recent1_popular.magazine_id==0].id

for i,t in magazine_dict:   
    cnt = 0
    for k , v in popular_read:
        if recent1_popular[recent1_popular.id==k].magazine_id == magazine_dict[int(i)]:
            print(recent1_popular[recent1_popular.id==k])
            cnt +=1
            if t/len(user_magazine) *20 <cnt:
                break
