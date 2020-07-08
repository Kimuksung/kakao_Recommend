# user에게 영화 추천 시스템
# user의 추천 목록을 보고 추천해준다.

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


rating_data=pd.read_csv("ratings.csv")
movie_data=pd.read_csv("movies.csv")
movie_data.info()

df_user_movie_ratings = rating_data.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

df_user_movie_ratings.head()
df_user_movie_ratings.shape

# matrix는 pivot_table 값을 numpy matrix로 만든 것 
matrix = np.array(df_user_movie_ratings)
matrix.shape #(671, 9066)
# user_ratings_mean은 사용자의 평균 평점 
user_ratings_mean = np.mean(matrix, axis = 1)
user_ratings_mean.shape # (671,)
# R_user_mean : 사용자-영화에 대해 사용자 평균 평점을 뺀 것.
matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)
matrix_user_mean.shape #(671, 9066)

pd.DataFrame(matrix_user_mean, columns = df_user_movie_ratings.columns).head()

U, sigma, Vt = svds(matrix_user_mean, k = 12)

print(U.shape)
print(sigma.shape)
print(Vt.shape)

sigma = np.diag(sigma)
sigma.shape

svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
svd_user_predicted_ratings.shape
df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns = df_user_movie_ratings.columns)
df_svd_preds.head()

sorted_user_predictions = df_svd_preds.iloc[300].sort_values(ascending=False)
user_data  = rating_data[rating_data.userId == 300]
recommendations = movie_data[~movie_data['movieId'].isin(user_data['movieId'])]
recommendations = recommendations.merge( pd.DataFrame(sorted_user_predictions).reset_index(), on = 'movieId')
recommendations = recommendations.rename(columns = {300: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:5, :]
