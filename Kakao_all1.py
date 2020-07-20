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

# --- 전처리 및 dataframe setting -----
def chainer(s):
    return list(chain.from_iterable(s.str.split(' ')))

def preprocessing_read(read):
    read_cnt_by_user = read['article_id'].str.split(' ').map(len)
    
    read_raw = pd.DataFrame({'dt': np.repeat(read['dt'], read_cnt_by_user),
                             'hr': np.repeat(read['hr'], read_cnt_by_user),
                             'user_id': np.repeat(read['user_id'], read_cnt_by_user),
                             'article_id': chainer(read['article_id'])})
    return read_raw.reset_index(drop=False , inplace=False).drop('index' ,axis=1)

def preprocessing_unix(metadata):
    tmp = []
    from datetime import datetime
    for i in range(len(metadata.reg_ts)):
        ts = int(metadata.reg_ts[i]/1000)
        tmp.append(datetime.utcfromtimestamp(ts).strftime('%Y%m%d'))
    
    metadata['date'] = tmp
    return metadata

def make_recent1_metadata(metadata , read_raw):
    recent_metadata = metadata[metadata.date.between('20190215' , '20190228')]    

    recent_read_popular = read_raw[read_raw.article_id.isin(recent_metadata.id)] # 930498
    
    popular_read_dict ={}
    for i in recent_read_popular['article_id']:
        if i in popular_read_dict.keys():
           popular_read_dict[i] += 1
        else:
            popular_read_dict[i] = 1
    
    popular_len = int(len(popular_read_dict) * 0.2) # 2073
    popular_read = sorted(popular_read_dict.items() , key = (lambda x:x[1]) , reverse =True)[:popular_len]
    tmp = []
    for i,j in popular_read:
        tmp.append(i)

    return metadata[metadata.id.isin(tmp)]

def flatten(A):
    rt = []
    for i in A:
        if isinstance(i,list): rt.extend(flatten(i))
        else: rt.append(i)
    return rt

def user_following_list(user_id):
    tmp2 = users[users.id==user_id].following_list
    tmp2 = tmp2.tolist()
    tmp2 = flatten(tmp2)
    return tmp2

def recent_same_readfollow(recent1_metadata , read):
    tmp = read[read.dt.between('20190222' , '20190228')]
    tmp_article_id = tmp[tmp.user_id == user_id].article_id
    list(recent1_metadata[recent1_metadata.id.isin(tmp_article_id)].user_id)
    
    answer = set(user_following_list) & set(list(recent1_metadata[recent1_metadata.id.isin(tmp_article_id)].user_id))    
    return recent1_metadata[recent1_metadata.user_id.isin(answer)].article_id

#doc2vec embedding을 이용한 작가 추천
def cleanText2(readData):
    text = re.sub('[^ \u3131-\u3163\uac00-\ud7a3]+','', readData)
    tmp=[]
    for i in text.split():
        if len(i)>1:
            tmp.append(i)
    text = " ".join(tmp)
    return text
  
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

def similar_writer( writer_id , cf_writer , model , tmp):      
    score = []
    for i in range(len(tmp)):
        score.append( (tmp.index[i], cosine_similarity(writer_id.reshape(-1, 128) , tmp.iloc[i].embedding.reshape(-1, 128))[0][0]))
    
    similar_writer_data = sorted( score , key = (lambda x:x[1]) , reverse=True)[:30]
    return similar_writer_data

def similar_writer_recom(user_following_list , writer , cf_writer , writer_df , recent1_metadata):
# 구독자들의 평균 vector값 구한다.
    tmp=[]
    for i in user_following_list:
        tmp.append(writer[i])
    
    tmp_embedding = np.array(tmp).mean(axis=0)
        
    similar_writer_recommend = similar_writer( tmp_embedding , cf_writer , model , writer_df)
    similar_writer_recommend = list(map(lambda x : x[0], similar_writer_recommend))
    
    return recent1_metadata[recent1_metadata.user_id.isin(similar_writer_recommend)].id

def cf_writer(metadata):
    metadata_emb = metadata[['user_id', 'title', 'keyword_list']]
    metadata_text2 = []
    
    for i in range( len(metadata_emb)):
        tmp = (metadata_emb.title[i]+" ".join(metadata_emb.keyword_list[i])).strip()
        tmp = cleanText2(tmp)
        metadata_text2.append(tmp)
        
    metadata_emb['text'] = metadata_text2
    cf_writer_tag = make_doc2vec_data(metadata_emb , 'text' , t_document = True)
    cf_writer = make_doc2vec_data(metadata_emb , 'text' )
    return cf_writer_tag , cf_writer

def recent_writer_recom(read , user_id ,recent1_metadata):
    a = read[read.dt.between('20190222', '20190228')]
    b = a[a.user_id==user_id]
    t = list(recent1_metadata[recent1_metadata.id.isin(set(b.article_id))].user_id)
    
    return similar_writer_recom(t , writer , cf_writer , writer_df , recent1_metadata)

#magazine based
def magazine_based(read , recent1_metadata) :
    a = read[read.dt.between('20190215','20190228')]
    b = set(a[a.user_id==user_id].article_id)
    t = list(recent1_metadata[recent1_metadata.id.isin(b)].magazine_id)
    
    tmp ={}
    for i in t:
        if i in tmp.keys():
            tmp[i] +=1
        else:
            tmp[i] = 1
    
    sort_value = sorted( tmp.items() , key=(lambda x:x[1]) , reverse=True)[:5]
    answer=[]
    if len(recent1_metadata[recent1_metadata.magazine_id==sort_value[0][0]].id)> 10:
        a = recent1_metadata[recent1_metadata.magazine_id==sort_value[0][0]].id
        b = read[read.article_id.isin(a)]
        t = list(b[b.dt.between('20190215' , '20190228')].article_id)
        tmp ={}
        for i in t:
            if i in tmp.keys():
                tmp[i] +=1
            else:
                tmp[i] = 1
    
        sort_value = sorted( tmp.items() , key=(lambda x:x[1]) , reverse=True)[:5]
        answer = list(map(lambda x : x[0], sort_value))
    
    else:
        answer = list(recent1_metadata[recent1_metadata.magazine_id==sort_value[0][0]].id)
    return answer

# recent2
def make_recent2_metadata(metadata ):
    recent2 = metadata[metadata.date.between('20190301' , '20190314')]
    # 구독자 수 세기
    follow_list ={}
    for i in users.following_list:
        for j in i:
            if j in follow_list.keys():
                follow_list[j] +=1
            else:
                follow_list[j] = 1
    
    tmp = []
    tmp2 = []
    tmp3 = []
    for i in recent2.user_id:
        if i in follow_list.keys():
            tmp.append(follow_list[i])
        else:
            tmp.append(0)
   
    magazine_dict ={}
    for i in magazine.magazine_tag_list:
        for j in i:
            if j in magazine_dict.keys():
                magazine_dict[j] +=1
            else:
                magazine_dict[j] = 1
        
    for i in recent2.magazine_id:
        a=0
        if i:
            for j in magazine[magazine.id == i].magazine_tag_list:
                for t in j:
                    a+= magazine_dict[t]
        tmp2.append(a)
        
    recent_pop = recent2.reset_index(drop=True)
    for i in range(0,len(tmp2)):
        tmp3.append(tmp2[i] + tmp[i])
    recent_pop['writer'] = tmp3

    recent_pop_num = int(len(recent_pop) * 0.2)
    return_data = recent_pop.sort_values(by=['writer'] , axis=0 , ascending=False)[:recent_pop_num].reset_index(drop=True)
    return return_data
#---- Main --------
#----popular + recentl based ---------
# 2019.02.22~2019.02.28
# popular = read num
metadata = preprocessing_unix(metadata)
read = preprocessing_read(read)

# 2019.02.15~2019.02.28 상위 20% view data에 대해서
recent1_metadata = make_recent1_metadata(metadata , read)

# 추천할 user_id
user_id = "#ae0aa864580107263a99c5087eb27a9d"

# ---- following based ------
# user의 구독 리스트
user_following_list = user_following_list(user_id)

# 1. 최근 2019.02.22~ 2019.02.28 user가 읽은 글 중에서 구독자와 일치하는 애들 우선 추천
recent_same_readfollow(recent1_metadata , read)

# 2. 인기있는 최근 글 중에서 구독자와 관련된 글 추천
recent1_metadata[recent1_metadata.user_id.isin(user_following_list)] # 16개

# 3. 구독자와 유사한 애들 추천

# 구독자끼리 상관관계가 높은 애들을 찾는다.
# 제목과 keyword를 이용하여 한글 data만 뽑아내어 작가 별로 128차원으로 임베딩 시켰다.
# 이를 이용하여 하나의 작가 당 모든 작가의 cos 유사도를 비교해서 상위 10만 뽑아내어 data로 저장
# -> 수정 why? 모든 작가와의 유사도를 구하려다 보니 하나당 1분 정도 소요되는데 19000개임으로 돌리기 무리라 판단
# -> 대안? 구독한 작가의 vector의 평균을 구하여 위와 비슷한 작가의 글을 추천

cf_writer_tag , cf_writer = cf_writer(metadata)
#make_doc2vec_models(cf_writer_tag , 'kakao_writer')
model = Doc2Vec.load('./kakao_writer_news_model.doc2vec')  

#writer = make_writer_embedding(metadata_emb ,model) #embedding modeling
#save_writer(writer) # data가 큼으로 embedding한 값 저장
writer = load_writer()
writer_df = pd.DataFrame(writer.items() , columns=['writer', 'embedding'])
writer_df = writer_df.set_index('writer')

similar_writer_recom(user_following_list , writer , cf_writer , writer_df , recent1_meta)

# 4. 최근 본글 작가와의 유사도(2019.02.22~2019.02.28)
recent_writer_recom(read , user_id ,recent1_metadata)

# ---magazine based ---
# 20190215 ~ 20190228 읽은 글 중에서 magazine의 비율을 보고 추천
magazine_based(read , recent1_metadata)

#----popular + recent2 based ---------
# 2019.03.01~2019.03.14
# read에 관한 data가 없기 때문에 popular 재정의 필요
# 앞의 다른 파이썬 파일을 보면
# magazine_tag_list / 구독자 수로 대체 가능
# user tag 대체 불가

# metadata를 통해 2019.03.01~03.14
recent2_metadata = make_recent2_metadata(metadata) #인기도 순으로 나열
# 1. 인기있는 최근 글 중에서 구독자와 관련된 글 추천
recent2_metadata[recent2_metadata.user_id.isin(user_following_list)]

# 2. 구독자와 유사한 애들 추천
cf_writer_tag , cf_writer = cf_writer(metadata)
#make_doc2vec_models(cf_writer_tag , 'kakao_writer')
model = Doc2Vec.load('./kakao_writer_news_model.doc2vec')  

#writer = make_writer_embedding(metadata_emb ,model) #embedding modeling
#save_writer(writer) # data가 큼으로 embedding한 값 저장
writer = load_writer()
writer_df = pd.DataFrame(list(writer.items()) , columns=['writer', 'embedding'])
writer_df = writer_df.set_index('writer')

similar_writer_recom(user_following_list , writer , cf_writer , writer_df , recent2_metadata)

# magazine based
# 3.1~3.14은 read data정보가 없기 때문에 어떤 magazine을 많이 읽었는지 알 수 가 없다.
# 따라서 이부분에 대해서는 모든 metadata를 이용하여 user가 어떤 magazine을 선호하는지 보고 이에 따라 추천해주는게 맞는거같다.

# All data
# 1. 자신이 읽은 글과 유사(~2019.03.14)
# read에 있지만 metadata에 없는 글도 있을 수 있다.
# 따라서 metadata에 없는 글은 생략하고 해야 할듯
def all_read_similar(read , metadata , user_id):
    tmp = list(read[read.user_id==user_id].article_id)
    list(metadata[metadata.id.isin(tmp)].index)
    temp=[]
    for i in list(metadata[metadata.id.isin(tmp)].index):
        temp.append(model.docvecs[i])
    temp=np.array(temp)
    read_mean = temp.mean(axis=0)
    
    score = []
    for i in range(len(metadata)):
        score.append( (i , cosine_similarity(read_mean.reshape(-1, 128) , model.docvecs[i].reshape(-1, 128))))
    
    similar_read_data = sorted( score , key = (lambda x:x[1]) , reverse=True)[:30]
    return similar_read_data

all_read_similar(read, metadata , user_id)

# 2. 구독자와 일치하는 거 중 많이 본거 추천(2019.02.28)
tmp = list(read[read.user_id==user_id].article_id)
set(list(metadata[metadata.id.isin(tmp)].user_id)) & set(user_following_list)


# 3. 같은 magazine에서 많이 본 거 추천(2019.02.28)
# 4. 검색어 tag를 이용하여 유사한 검색어 추천(2019.02.28)





