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

def chainer(s):
    return list(chain.from_iterable(s.str.split(' ')))

read_cnt_by_user = read['article_id'].str.split(' ').map(len)

read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                         'hr': np.repeat(read['hr'], read_cnt_by_user),
                         'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                         'article_id': chainer(read['article_id'])})

# metadata 2018.10 ~ 2019.3.14
# data preprocessing unix data -> real data
def trans_unix(metadata):
    tmp = []
    from datetime import datetime
    for i in range(len(metadata.reg_ts)):
        ts = int(metadata.reg_ts[i]/1000)
        tmp.append(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
    
    metadata['date'] = tmp
    metadata.head()
    metadata = metadata.drop('reg_ts' , axis=1)
    return metadata

# recent2 대해 수행 중
# metadata를 통해 2019.03.01~03.14
def make_recent2_metadata(metadata ):
    metadata = trans_unix(metadata)
    #metadata 원하는 기간 데이터만 추출
    metadata = metadata[metadata.date.between('20190301' , '20190314')]
    
    return metadata

recent2 = make_recent2_metadata(metadata)

'''
# 구독자 수 비교 인기있는 애들 vs 전체 애들
recent1_popular
metadata[metadata.date.between('20190215' , '20190228')]

follow_list ={}
for i in users.following_list:
    for j in i:
        if j in follow_list.keys():
            follow_list[j] +=1
        else:
            follow_list[j] = 1

pop_kudok = 0
for i in recent1_popular.user_id:
    if i in follow_list.keys():
        pop_kudok += follow_list[i]
pop_kudok/len(recent1_popular.user_id) # 1374.227689339122

normal_kudok = 0
for i in metadata[metadata.date.between('20190215' , '20190228')].user_id:
    if i in follow_list.keys():
        normal_kudok += follow_list[i]

normal_kudok/len(metadata[metadata.date.between('20190215' , '20190228')]) # 372.9028273401043
'''

# 구독자 수 세기
follow_list ={}
for i in users.following_list:
    for j in i:
        if j in follow_list.keys():
            follow_list[j] +=1
        else:
            follow_list[j] = 1

tmp = []
for i in recent2.user_id:
    if i in follow_list.keys():
        tmp.append(follow_list[i])
    else:
        tmp.append(0)

'''
recent2['writer_num'] = tmp
sorted(follow_list.items() , key = (lambda x:x[1]) , reverse =True)[:10] # 0 ~ 292413
'''
magazine_dict ={}
for i in magazine.magazine_tag_list:
    for j in i:
        if j in magazine_dict.keys():
            magazine_dict[j] +=1
        else:
            magazine_dict[j] = 1

tmp2 = []
for i in recent2.magazine_id:
    a=0
    if i:
        for j in magazine[magazine.id == i].magazine_tag_list:
            for t in j:
                a+= magazine_dict[t]
    tmp2.append(a)
       
recent_pop = recent2.reset_index(drop=True)
tmp3 = []
for i in range(0,len(tmp2)):
    tmp3.append(tmp2[i] + tmp[i])
recent_pop['writer'] = tmp3
recent_pop.info()

recent_pop_num = int(len(recent_pop) * 0.2)
recent_pop.sort_values(by=['writer'] , axis=0 , ascending=False)[:recent_pop_num]

len(set(metadata.user_id))













