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

# recent1 대해 수행 중
# metadata를 통해 2019.02.15~02.28
def make_recent1_metadata(metadata , read_raw):
    metadata = trans_unix(metadata)
    #metadata 원하는 기간 데이터만 추출
    metadata = metadata[metadata.date.between('20181001' , '20190314')]
    
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

def cleanText(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData) 
    text = text.strip()
    text = re.sub('\xa0', '', text) 
    return text

def konlpy(readData):
    from konlpy.tag import Kkma
    kkma = Kkma()
    tmp = []
    for i in readData:
        ex_pos = kkma.pos(i)        
        # NNG 일반 명사 NNP고유 명사 NP 대명사
        nouns = []        
        for word , wclass in ex_pos:
            if wclass == "NNG" or wclass =="NNP" or wclass =="NP":
                nouns.append(word)
                
        tmp.append(" ".join(nouns))
    return tmp        

recent1_popular , recent_read_popular = make_recent1_metadata(metadata , read_raw)
# 안하는게 더 나아보임
recent1_cf = []
for i in range(0 , len(recent1_popular)):
    recent1_cf.append(cleanText(recent1_popular.title.iloc[i]) + " " +cleanText(recent1_popular.sub_title.iloc[i]))
    
recent1_popular['text'] = recent1_cf
recent1_popular_cf = recent1_popular[['id' , 'text']]

def make_doc2vec_models(tagged_data, name, vector_size=128, window = 3, epochs = 40, min_count = 0, workers = 4):
    model = Doc2Vec(tagged_data, vector_size=vector_size, window=window, epochs=epochs, min_count=min_count, workers=workers)
    model.save(f'./{name}_news_model.doc2vec')
    
def make_doc2vec_data(data, column, t_document=False):
    data_doc = []
    for tag, doc in zip(data.index, data[column]):
        doc = doc.split(" ")
        data_doc.append(([tag], doc))
    if t_document:
        data = [TaggedDocument(words=text, tags=tag) for tag, text in data_doc]
        return data
    else:
        return data_doc    

def make_user_embedding(user_id , recent1_popular_cf ,model):       
    user_read = recent_read_popular[recent_read_popular.user_id == user_id].article_id
    user_embedding = []   
    for i in set(user_read):
        user_embedding.append(model.docvecs[recent1_popular_cf[recent1_popular_cf.id==i].index[0]])
    user_embedding = np.array(user_embedding)
    user = np.mean(user_embedding , axis=0)
    
    return user    

def get_recommened_contents(user, data_doc, model , recent1_popular_cf , n):
    scores = []

    for tags, text in data_doc:
        trained_doc_vec = model.docvecs[tags[0]]
        scores.append(cosine_similarity(user.reshape(-1, 128), trained_doc_vec.reshape(-1, 128)))

    scores = np.array(scores).reshape(-1)
    scores = np.argsort(-scores)[:n]
    
    return recent1_popular_cf.loc[scores, :]
    
user_id = "#0c04b1be93b3c76750c46161c7990bfb"    
#user_id = "#507a2c092bee4c7d9332d83c70a00e6b"       
recent1_popular_cf = recent1_popular_cf.reset_index(drop=True)    
cf_tag = make_doc2vec_data(recent1_popular_cf , 'text' , t_document = True)
cf = make_doc2vec_data(recent1_popular_cf , 'text' )       
make_doc2vec_models(cf_tag , 'kakao_recent1_cf')
model = Doc2Vec.load('./kakao_recent1_cf_news_model.doc2vec')    

user = make_user_embedding(user_id , recent1_popular_cf , model)         
get_recommened_contents(user , cf , model , recent1_popular_cf , 10).text    
    
    
for i in set(recent_read_popular[recent_read_popular.user_id == user_id].article_id):
    print(recent1_popular[recent1_popular.id==i].text)
   
    
recent1_popular_cf.iloc[745].text
recent1_popular_cf.iloc[1186].text
recent1_popular_cf.iloc[495].text
recent1_popular_cf.iloc[1529].text
recent1_popular_cf.iloc[910].text






    