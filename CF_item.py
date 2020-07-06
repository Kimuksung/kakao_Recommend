# Collaborate filtering
# item based

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rating_data=pd.read_csv("ratings.csv")
movie_data=pd.read_csv("movies.csv")
rating_data.drop('timestamp', axis = 1, inplace=True)
rating_data.head(2)
movie_data.head(2)

user_movie_rating = pd.merge(rating_data, movie_data, on = 'movieId')
user_movie_rating.head(2)

movie_user_rating = user_movie_rating.pivot_table('rating', index = 'title', columns='userId')
movie_user_rating.head()

movie_user_rating.fillna(0, inplace = True)
movie_user_rating.head(3)

item_based_collabor = cosine_similarity(movie_user_rating)
item_based_collabor.shape

item_based_collabor = pd.DataFrame(data = item_based_collabor, index = movie_user_rating.index, columns = movie_user_rating.index)
item_based_collabor

item_based_collabor['Godfather, The (1972)'].sort_values(ascending=False)[:6]
