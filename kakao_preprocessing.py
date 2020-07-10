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

# 쓸모 없는 데이터 제거
def cleanText(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text

# 10000개의 metadata로만 먼저 실험
# Data가 너무 많아 일부만 해보기로 함
# Doc = 제목 + 부제목 + keyword tag 3개로 묶어 Doc2Vec화 size = 128개로 함
    
tmp = metadata[:10000]
temp = []
for i in range(tmp.shape[0]):
    temp.append(cleanText(tmp['title'][i]+" "+tmp['sub_title'][i]+" "+str(tmp['keyword_list'][i])))

def make_doc2vec_models(tagged_data, name, vector_size=128, window = 3, epochs = 40, min_count = 0, workers = 4):
    model = Doc2Vec(tagged_data, vector_size=vector_size, window=window, epochs=epochs, min_count=min_count, workers=workers)
    model.save(f'./{name}_model.doc2vec')

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

def make_user_embedding(user_id, tmp , model):
    user_article = df[df.user_id == user_id]['article_id']
    
    answer_index = []
    user_embedding = []
    for i in set.intersection(*map(set, [tmp['id'], user_article])):
        answer_index.append(tmp[tmp.id==i].index[0])
    for i in answer_index:
        print(tmp.iloc[i])
        user_embedding.append(model.docvecs[i])
    user_embedding = np.array(user_embedding)
    return np.mean(user_embedding, axis = 0)

tmp = pd.DataFrame({"id" : tmp.id , "text": temp})
data_content_tag = make_doc2vec_data(tmp, 'text' , t_document = True)
data_content = make_doc2vec_data(tmp, 'text')

make_doc2vec_models(data_content_tag, name="kakao")
model_content = Doc2Vec.load('./kakao_model.doc2vec')

user = make_user_embedding("#38223f88af26b635b9d3d39c615a0219" , tmp , model_content)

def get_recommened_contents(user, data_doc, model):
    scores = []

    for tags, text in data_doc:
        trained_doc_vec = model.docvecs[tags[0]]
        scores.append(cosine_similarity(user.reshape(-1, 128), trained_doc_vec.reshape(-1, 128)))

    scores = np.array(scores).reshape(-1)
    scores = np.argsort(-scores)[:5]
    
    return df.loc[scores, :]

print("-"*30)
get_recommened_contents(user , data_content , model_content)
