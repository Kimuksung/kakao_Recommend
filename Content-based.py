# content-based-filtering

import pandas as pd
import numpy as np
import ast

movie = pd.read_csv("movies_metadata.csv")
movie.info()

movie.release_date
movie.id
#movie.keywords
movie.tagline

data = movie[['id', 'genres' , 'vote_average' , 'vote_count' , 'popularity' , 'title', 'overview' , 'release_date']]

data.info()

m = data['vote_count'].quantile(0.9)
data = data.loc[data['vote_count'] >= m ]

c = data['vote_average'].mean()

print(m) # 160.0
print(c) # 6.473589462129529

def weighted_rating(x, m=m, c=c):
    v = x['vote_count']
    R = x['vote_average']
    
    return ( v / (v+m) * R) + (m/ (m+v) * c)

data['score'] = data.apply(weighted_rating , axis = 1)
data.info()

movie.shape
data.shape

data['genres'].head(5)
data['genres'] = data['genres'].apply(ast.literal_eval)
data['genres'] = data['genres'].apply(lambda x : [d['name'] for d in x]).apply(lambda x:" ".join(x))
from sklearn.feature_extraction.text import CountVectorizer
counter_vector = CountVectorizer(ngram_range = (1,3))
counter_genres = counter_vector.fit_transform(data['genres'])
counter_genres.shape

from sklearn.metrics.pairwise import cosine_similarity
genres_cos_sim = cosine_similarity(counter_genres , counter_genres).argsort()[:,::-1]
genres_cos_sim.shape # (4555, 4555)

data2 = data.reset_index()

movie_title = "The Dark Knight Rises"
top=30
target_movie = data2[data2['title']==movie_title].index.values
sim_index = genres_cos_sim[target_movie , :top].reshape(-1)
sim_index = sim_index[sim_index != target_movie]
result = data2.iloc[sim_index].sort_values('score' , ascending=False)
result.info()
result=result.drop(['index','id','release_date'] , axis=1)
result
