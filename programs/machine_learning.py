import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import nltk
import collections

# Read the IMDb movies data into a pandas DataFrame
imdb_movies = pd.read_csv('data/imdb_movies.csv', index_col="names")


# Set the feature names and target names
feature_names = ['month_num', 'orig_lang_num', 'genre_num', 'country_num', 'budget_x']
target_names = ['revenue', 'score']

# Extract the feature values (X) and target values (y) from the IMDb movies data
imdb_x_vals = imdb_movies[feature_names].values.astype('float')
imdb_y_vals = imdb_movies[target_names].values

# Create a dictionary to map actors to frequencies
actor_to_freq = collections.defaultdict(int)
for crew in imdb_movies['crew'].dropna().str.split(','):
    for actor in crew:
        actor_to_freq[actor] += 1

# Convert the actors to frequencies
actor_freqs = []
for crew in imdb_movies['crew'].dropna():
    actor_freq_list = [actor_to_freq[actor] for actor in crew.split(',')]
    actor_freqs.append(actor_freq_list)

# Add the actor frequencies to the feature set
imdb_x_vals = np.concatenate((imdb_x_vals, np.array(actor_freqs).reshape(-1, 1)), axis=1)

# Split the data into a training set and a test set
train_x_vals, test_x_vals, train_y_vals, test_y_vals = train_test_split(
    imdb_x_vals, imdb_y_vals, test_size=0.25
)

# Create a neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(train_x_vals, train_y_vals, epochs=100)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(test_x_vals, test_y_vals)
print('Test loss:', test_loss)
print('Test mae:', test_mae)

# Predict the target values for the test set
predicted_values = model.predict(test_x_vals)

# Extract the predicted revenue and score from the predicted values
predicted_revenue = predicted_values[:, 0]
predicted_score = predicted_values[:, 1]

# Calculate the root mean squared error (RMSE) and R-squared
rmse = np.sqrt(mean_squared_error(test_y_vals, predicted_values))
r2 = r2_score(test_y_vals, predicted_values)

print('RMSE:', rmse)
print('R-squared:', r2)
