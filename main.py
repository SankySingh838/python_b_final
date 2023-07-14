import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

imdb_movies = pd.read_csv('data/stripped_imdb_movies_data.csv', index_col="names")

# Set the feature names
feature_names = ['month_num', 'genre_num', 'budget_x', 'country_num']
target_names = ['revenue', 'score']

imdb_x_vals = imdb_movies[feature_names].values
imdb_y_vals = imdb_movies[target_names].values

regr = LinearRegression()
regr.fit(imdb_x_vals, imdb_y_vals)

# Example of using the trained model on new data
new_data = pd.read_csv('data/new_movie_data.csv', index_col="names")
new_data = new_data[feature_names]  # Keep only the desired features from the new data

predicted_values = regr.predict(new_data)
print("Predicted values for new data:")
print(predicted_values)
