import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read the IMDb movies data into a pandas DataFrame
imdb_movies = pd.read_csv('../data/stripped_imdb_movies_data.csv', index_col="names")

# Set the feature names and target names
feature_names = ['month_num', 'genre_num', 'budget_x', 'country_num']
target_names = ['revenue', 'score']

# Extract the feature values (X) and target values (y) from the IMDb movies data
imdb_x_vals = imdb_movies[feature_names].values
imdb_y_vals = imdb_movies[target_names].values

# Create a Linear Regression model and fit it to the IMDb movies data
regr = LinearRegression()
regr.fit(imdb_x_vals, imdb_y_vals)

# Load the new movie data for prediction
new_data = pd.read_csv('../data/new_movie_data.csv', index_col="names")
new_data = new_data[feature_names]  # Keep only the desired features from the new data

# Predict the target values (revenue and score) for the new movie data using the trained model
predicted_values = regr.predict(new_data)
print("Predicted values for new data:")
print(predicted_values)

# Extract the predicted revenue and score from the predicted values
predicted_revenue = predicted_values[:, 0]
predicted_score = predicted_values[:, 1]

# Create a list of movie names
movie_names = new_data.index.values

# Plot the predicted revenue for the new movies
plt.figure(figsize=(10, 6))
plt.bar(movie_names, predicted_revenue)
plt.xlabel('Movie')
plt.ylabel('Predicted Revenue')
plt.title('Predicted Revenue for New Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predicted_revenue.png')  # Save the plot as an image
plt.show()

# Plot the predicted score for the new movies
plt.figure(figsize=(10, 6))
plt.bar(movie_names, predicted_score)
plt.xlabel('Movie')
plt.ylabel('Predicted Score')
plt.title('Predicted Score for New Movies')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('predicted_score.png')  # Save the plot as an image
plt.show()