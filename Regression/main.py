# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    bike_data = pd.read_csv('daily-bike-share.csv')
    bike_data['day'] = pd.DatetimeIndex(bike_data['dteday']).day
    numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'day']
    bike_data[numeric_features + ['rentals']].describe()
    print(bike_data.head())

    # Separate features and labels
    # After separating the dataset, we now have numpy arrays named **X** containing the features, and **y** containing the labels.
    X, y = bike_data[['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum',
                      'windspeed']].values, bike_data['rentals'].values

    # Split data 70%-30% into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    print('Training Set: %d rows\nTest Set: %d rows' % (X_train.shape[0], X_test.shape[0]))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
