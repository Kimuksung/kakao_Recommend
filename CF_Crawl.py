import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

# 크롤링 
def cleanText(readData):
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData) 
    return text

def crawl(url , name):
    html = urlopen(url)
    source = html.read() # 바이트코드 type으로 소스를 읽는다.
    html.close()
    soup = BeautifulSoup(source, "html5lib")
    
    category=[] 
    for node in soup.findAll(class_=name):
        data = cleanText("".join(node.findAll(text=True)))
        print(data)
        category.append(data)
    return category

category1 = crawl("https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=101" , "cluster_text_headline nclicks(cls_eco.clsart)")
category2 = crawl("https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=102" , "cluster_text_headline nclicks(cls_nav.clsart)")
category3 = crawl("https://news.naver.com/main/main.nhn?mode=LSD&mid=shm&sid1=105" , "cluster_text_headline nclicks(cls_sci.clsart)")

len(category1) # 18
len(category2) # 24
len(category3) # 12

# data
category = [1]*len(category1)
df1 = pd.DataFrame({"category":category , "content":category1})
category = [2]*len(category2)
df2 = pd.DataFrame({"category":category , "content":category2})
category = [3]*len(category3)
df3 = pd.DataFrame({"category":category , "content":category3})

df = pd.concat([df1 , df2 , df3] , ignore_index = True)

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

data_content_tag = make_doc2vec_data(df, 'content' , t_document = True)
data_content = make_doc2vec_data(df, 'content')

make_doc2vec_models(data_content_tag, name="content")
model_content = Doc2Vec.load('./content_news_model.doc2vec')

user_category_1 = df.loc[random.sample(df.loc[df.category == 1, :].index.values.tolist(), 5), :]  #경제
user_category_2 = df.loc[random.sample(df.loc[df.category == 2, :].index.values.tolist(), 5), :]  #사회
user_category_3 = df.loc[random.sample(df.loc[df.category == 3, :].index.values.tolist(), 5), :]  #IT

def make_user_embedding(index_list, data_doc, model):
    user = []
    user_embedding = []
    for i in index_list:
        user.append(data_doc[i][0][0])
    for i in user:
        user_embedding.append(model.docvecs[i])        
    user_embedding = np.array(user_embedding)
    user = np.mean(user_embedding, axis = 0)
    return user

user_1 = make_user_embedding(user_category_1.index.values.tolist(), data_content, model_content)
user_2 = make_user_embedding(user_category_2.index.values.tolist(), data_content, model_content)
user_3 = make_user_embedding(user_category_3.index.values.tolist(), data_content, model_content)
user_1.shape

def get_recommened_contents(user, data_doc, model):
    scores = []

    for tags, text in data_doc:
        trained_doc_vec = model.docvecs[tags[0]]
        scores.append(cosine_similarity(user.reshape(-1, 128), trained_doc_vec.reshape(-1, 128)))

    scores = np.array(scores).reshape(-1)
    scores = np.argsort(-scores)[:5]
    
    return df.loc[scores, :]

result = get_recommened_contents(user_1, data_content, model_content)
pd.DataFrame(result.loc[:, ['category', 'title_content']])






