import pandas as pd
import numpy as np
from sklearn import linear_model

imdb_movies = pd.read_csv('data/imdb_movies.csv', index_col="names")

# Set the feature names
imdb_x_vals = imdb_movies[['month_num', 'genre_num', 'budget_x', 'country_num']]
imdb_y_vals = imdb_movies[['revenue', 'score']]

regr = linear_model.LinearRegression()
regr.fit(imdb_x_vals, imdb_y_vals)


for index, row in imdb_movies.iterrows():
    genre_num_movie = row['genre_num']
    month_num_movie = row['month_num']
    country_num_movie = row['country_num']
    budget_x_movie = row['budget_x']
    predicted_shit = regr.predict([[month_num_movie, genre_num_movie, budget_x_movie, country_num_movie]])
