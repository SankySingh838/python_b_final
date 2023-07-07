import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns

imdb_movies = pd.read_csv('imdb_movies.csv', index_col = "names")

imdb_x_vals = imdb_movies[['month_num', 'genre_num', 'budget_x', 'country_num']]
imdb_y_vals = imdb_movies[['revenue', 'score']]

regr = linear_model.LinearRegression()
regr.fit(imdb_x_vals, imdb_y_vals)
predicted_shit = regr.predict([[3, 1, 75000000, 1]])
print(predicted_shit)
# date, score, genre, orig_lang, first budget_x, countryasdkl;fawe;krfajl;sdfj badri likes cock



