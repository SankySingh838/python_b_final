import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Read the IMDb movies data into a pandas DataFrame
imdb_movies = pd.read_csv('data/training_data_imdb.csv', index_col="names")

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
new_data = pd.read_csv('data/testing_data.csv', index_col="names")
print(new_data)
new_data = new_data[feature_names]  # Keep only the desired features from the new data

# Predict the target values (revenue and score) for the new movie data using the trained model
predicted_values = regr.predict(new_data)
print("Predicted values for new data:")
print(predicted_values)

# Read the CSV file into a DataFrame
testing_answer_csv = pd.read_csv('data/testing_data_imdb_answers.csv')

# Extract the 'score' and 'revenue' columns
score = testing_answer_csv['score'].values
revenue = testing_answer_csv['revenue'].values

# Create a NumPy array with revenue in the first column and score in the second column
testing_answer = np.column_stack((revenue, score))
print("Testing Answers:")
print(testing_answer)
# Extract the predicted revenue and score from the predicted values
predicted_revenue = predicted_values[:, 0]
predicted_score = predicted_values[:, 1]

rmse = np.sqrt(mean_squared_error(testing_answer, predicted_values))
print("Root Mean Squared Error:", rmse)


r2 = r2_score(testing_answer, predicted_values)
print("R-squared:", r2)


# Plot Actual vs. Predicted Revenue
plt.scatter(testing_answer[:, 0], predicted_revenue)
plt.plot(testing_answer[:, 0], testing_answer[:, 0], color='red')  # Add y = x line
plt.xlabel('Actual Revenue')
plt.ylabel('Predicted Revenue')
plt.title('Actual vs. Predicted Revenue')
plt.savefig('actual_vs_predicted_revenue.png')

# Plot Actual vs. Predicted Score
plt.scatter(testing_answer[:, 1], predicted_score)
plt.plot(testing_answer[:, 1], testing_answer[:, 1], color='red')  # Add y = x line
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title('Actual vs. Predicted Score')
plt.savefig('actual_vs_predicted_score.png')


