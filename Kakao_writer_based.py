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

def cleanText(readData):    
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》_Ⅰ\"}]', '', readData)
    text = re.sub('[\d]', '', text)
    #text = re.sub('[a-zA-Z]', '', text)
    #text = re.sub(r'(?:\b[0-9a-zA-Zㄱ-ㅎㅏ-ㅣ]\b|[?!\W]+)\s*', ' ', text)   
    text = re.sub('([ㄱ-ㅎㅏ-ㅣ]+)'  , repl='', string=text).strip()
    tmp=[]
    for i in text.split():
        if len(i)>1:
            tmp.append(i)
    text = " ".join(tmp)
    return text

def cleanText2(readData):
    text = re.sub('[^ \u3131-\u3163\uac00-\ud7a3]+','', readData)
    tmp=[]
    for i in text.split():
        if len(i)>1:
            tmp.append(i)
    text = " ".join(tmp)
    return text

# data 전처리
# 2가지로 나누어서 처리
# 1. 숫자/한글자/특수문자 제거
# 2. 한글 및 한글자 이상만 추출
# 2번 선택
metadata_emb = metadata[['user_id', 'title', 'keyword_list']]
" ".join(metadata_emb.keyword_list[0])
metadata_text = []
for i in range( len(metadata_emb)):
    tmp = (metadata_emb.title[i]+" ".join(metadata_emb.keyword_list[i])).strip()
    tmp = cleanText(tmp)
    metadata_text.append(tmp)

metadata_text2 = []
for i in range( len(metadata_emb)):
    tmp = (metadata_emb.title[i]+" ".join(metadata_emb.keyword_list[i])).strip()
    tmp = cleanText2(tmp)
    metadata_text2.append(tmp)


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
    
def make_writer_embedding( metadata_emb ,model):       
    writer = set(metadata_emb.user_id)
    cnt = 0
    writer_embedding ={}
    for i in writer:
        print(cnt)
        writer_temp = []
        for j in metadata_emb[metadata_emb.user_id==i].index:
            writer_temp.append(model.docvecs[j])
        writer_embedding[i] = np.array(writer_temp).mean(axis=0)    
        cnt +=1
    return writer_embedding  

def save_writer(writer):
    import pickle
    with open('writer.pickle', 'wb') as f:
        pickle.dump(writer, f, pickle.HIGHEST_PROTOCOL)
        print("Complete save pickle")
        
def load_writer():
    import pickle
    with open('writer.pickle', 'rb') as f:
        sample = pickle.load(f)
    return sample

def similar_writer( writer_id , cf_writer , model , similar_writer_df):
    tmp = similar_writer_df
    
    score = []
    for i in range(len(tmp)):
        score.append( (tmp.index[i], cosine_similarity(tmp.loc[writer_id].embedding.reshape(-1, 128) , tmp.iloc[i].embedding.reshape(-1, 128))[0][0]))
    
    similar_writer_data = sorted( score , key = (lambda x:x[1]) , reverse=True)[1:10]
    return similar_writer_data

metadata_emb['text'] = metadata_text2
cf_writer_tag = make_doc2vec_data(metadata_emb , 'text' , t_document = True)
cf_writer = make_doc2vec_data(metadata_emb , 'text' )   

make_doc2vec_models(cf_writer_tag , 'kakao_writer')
model = Doc2Vec.load('./kakao_writer_news_model.doc2vec')  

writer = make_writer_embedding(metadata_emb ,model) #embedding modeling
save_writer(writer) # data가 큼으로 embedding한 값 저장
    
similar_writer_df = pd.DataFrame(writer.items() , columns=['writer', 'embedding'])
similar_writer_df = similar_writer_df.set_index('writer')
similar_writer( '@chungsana' , cf_writer , model , similar_writer_df)
#feedback하자면 함수에 모르고 2번 반복시켜 돌렸다가 매우 큰 시간 손해를 봄

similar_writer_all={}
cnt=0
for i in set(metadata.user_id): # 19065
    tmp = []
    cnt +=1
    print(cnt)
    for j in similar_writer( i , cf_writer , model , similar_writer_df):
        tmp.append(j[0])
    similar_writer_all[i] = tmp

