import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

imdb_movies = pd.read_csv('data/imdb_movies.csv', index_col="names")

# Set the feature names
feature_names = ['month_num', 'genre_num', 'budget_x', 'country_num']
target_names = ['revenue', 'score']

imdb_x_vals = imdb_movies[feature_names].values
imdb_y_vals = imdb_movies[target_names].values

regr = LinearRegression()
regr.fit(imdb_x_vals, imdb_y_vals)

# Example of using the trained model on new data
#{'month_num': [3, 6, 9], 'genre_num': [2, 1, 4], 'budget_x': [5000000, 10000000, 20000000], 'country_num': [1, 3, 2]}
new_data = pd.read_csv('data/new_movie_dataa.csv', index_col="names")
predicted_values = regr.predict(new_data)
print("Predicted values for new data:")
print(predicted_values)
