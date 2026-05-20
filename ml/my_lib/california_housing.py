from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from models_utils import LinearRegression as MyLinReg
from sklearn.linear_model import LinearRegression as SklearnLinReg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from termcolor import colored

np.random.seed(42)

dataset = fetch_california_housing()
df = pd.DataFrame(data = dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

my_model = MyLinReg()
sk_model = SklearnLinReg()

my_model.fit(X_train, y_train)
sk_model.fit(X_train, y_train)

print(colored('MY_MODEL', 'red'))
my_model_y_pred = my_model.predict(X_test)
my_model_mse = mean_squared_error(y_test, my_model_y_pred)
my_model_rmse = np.sqrt(my_model_mse)
my_model_mae = mean_absolute_error(y_test, my_model_y_pred)
my_model_r2 = r2_score(y_test, my_model_y_pred)
print(colored(f'MSE = {my_model_mse:.2}\nRMSE = {my_model_rmse:.2}\nMAE = {my_model_mae:.2}\nR2 = {my_model_r2:.2}', 'green'))

print(colored('SK_MODEL', 'red'))
sk_model_y_pred = sk_model.predict(X_test)
sk_model_mse = mean_squared_error(y_test, sk_model_y_pred)
sk_model_rmse = np.sqrt(sk_model_mse)
sk_model_mae = mean_absolute_error(y_test, sk_model_y_pred)
sk_model_r2 = r2_score(y_test, sk_model_y_pred)
print(colored(f'MSE = {sk_model_mse:.2}\nRMSE = {sk_model_rmse:.2}\nMAE = {sk_model_mae:.2}\nR2 = {sk_model_r2:.2}', 'green'))
