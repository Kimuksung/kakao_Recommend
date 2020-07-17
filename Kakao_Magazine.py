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
def make_recent1_metadata(metadata , read_raw):
    recent_metadata = metadata[metadata.date.between('20190215' , '20190228')]
    recent_metadata.id
    
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

    return metadata[metadata.id.isin(tmp)] , recent_read_popular[recent_read_popular.article_id.isin(tmp)]

#아이디어는 2개
# 1. magazine에 대해서 user가 읽은 거에 대해 magazine별로 비율을 나누어 추천(인기도 높고 최신 순으로 추천할듯)
# 2. magazine에 대해 28000개 labeling에 대해 embedding 시키고 이를 이용하여 
# user가 magazine어떤 분야에 흥미가 있는지를 나타낼수있다.  

# 아이디어 1.
def recent1_magazine_based(user_id , recent1_popular , recent_read_popular  ):
    user_article = recent_read_popular[recent_read_popular.user_id==user_id].article_id
    user_article = list(set(user_article))
    
    magazine_dict ={}
    user_magazine = recent1_popular[recent1_popular.id.isin(user_article)].magazine_id
    for i in user_magazine:
        if i in magazine_dict.keys():
            magazine_dict[i] +=1
        else:
            magazine_dict[i] = 1
    
    # 인기도 순으로 나타내어 제일 많은 magazine의 id를 보고 상위 몇개만 추천
    magazine_dict = sorted(magazine_dict.items(), key = (lambda x:x[1]) , reverse =True)
    magazine_pop_id , magazine_pop_len = magazine_dict[0]
    magazine_recommend_id = list(recent1_popular[recent1_popular.magazine_id==magazine_pop_id].id)
    len(magazine_recommend_id)
    
    magazine_dict2 ={}
    read_raw_list = list(recent_read_popular['article_id'])
    for i in magazine_recommend_id:
        magazine_dict2[i] = read_raw_list.count(i)
        
    magazine_dict2 = sorted(magazine_dict2.items(), key = (lambda x:x[1]) , reverse =True)
    return magazine_dict2

recent1_popular , recent_read_popular = make_recent1_metadata(metadata , read_raw)
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
magazine_based = recent1_magazine_based( user_id , recent1_popular , recent_read_popular)


# magazine에 대해 2019.03.01 ~ 2019.03.14 에 대해 인기도를 대체 할 수 있는가?
# 어느 정도 대체 할 수 있을걸로 보인다.
# 인기글 1555.1650893796004 전체 글 493.7826882605911
magazine.magazine_tag_list

recent1_popular
metadata[metadata.date.between('20190215' , '20190228')]

magazine_dict ={}
for i in magazine.magazine_tag_list:
    for j in i:
        if j in magazine_dict.keys():
            magazine_dict[j] +=1
        else:
            magazine_dict[j] = 1
magazine_pop = 0   # 1478962
for i in magazine[magazine.id.isin(recent1_popular.magazine_id)].magazine_tag_list:
    for j in i:
        magazine_pop += magazine_dict[j]
magazine_pop/951 # 1555.1650893796004

magazine_normal = 0   # 5396551
for i in magazine[magazine.id.isin(metadata[metadata.date.between('20190215' , '20190228')].magazine_id)].magazine_tag_list:
    for j in i:
        magazine_normal += magazine_dict[j]
magazine_normal/10929 # 493.7826882605911

# user의 tag
user_tag_dict ={}
for i in users.keyword_list:
    if i:
        for j in i:
            for t in j['keyword'].split(' '):
                if t in user_tag_dict.keys():
                    user_tag_dict[t] +=1
                else :
                    user_tag_dict[t] = 1

user_pop_tag = 0        
for i in users[users.id.isin(read_raw[read_raw.article_id.isin(recent1_popular.id)].user_id)].keyword_list:
    if i:
        for j in i:
            for t in j['keyword'].split(' '):
                user_pop_tag += user_tag_dict[t]
#175801705
            
user_pop_tag/59236 # 2967.8186406914715

user_normal_tag = 0
for i in users[users.id.isin(read_raw[read_raw.article_id.isin(metadata[metadata.date.between('20190215' , '20190228')].id)].user_id)].keyword_list:
    if i:
        for j in i:
            for t in j['keyword'].split(' '):
                user_normal_tag += user_tag_dict[t]

# 181277946
user_normal_tag/63218 #2867.505235850549
                
                