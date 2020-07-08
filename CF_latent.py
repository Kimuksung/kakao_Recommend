# Collaborate filtering
# item based
# 영화와 유사한 영화 찾
from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


rating_data=pd.read_csv("ratings.csv")
movie_data=pd.read_csv("movies.csv")

rating_data.head()

print(movie_data.shape)
print(rating_data.shape)

rating_data.drop('timestamp', axis = 1, inplace = True)
movie_data.drop('genres', axis = 1, inplace = True)

user_movie_data = pd.merge(rating_data, movie_data, on = 'movieId')
user_movie_data.head()
user_movie_data.shape

user_movie_rating = user_movie_data.pivot_table('rating', index = 'userId', columns='title').fillna(0)
user_movie_rating.shape

movie_user_rating = user_movie_rating.values.T
movie_user_rating.shape

SVD = TruncatedSVD(n_components=12)
matrix = SVD.fit_transform(movie_user_rating)
matrix.shape
matrix

corr =np.corrcoef(matrix)
corr

movie_title = user_movie_rating.columns
movie_title_list = list(movie_title)
coffey_hands = movie_title_list.index("Guardians of the Galaxy (2014)")

choose = corr[3405]
list(movie_title[(choose >= 0.9)])[:50]
