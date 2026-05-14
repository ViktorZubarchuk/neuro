import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from termcolor import colored
from calc_price import calc_price

RS = 42

models = {
    'lin_reg': (
        LinearRegression(), 
        {}
    ),
    'ridge': (
        Ridge(),
        {}
    ),
    'rand_forest': (
        RandomForestRegressor(random_state=RS),
        {
        'model__n_estimators': [200, 250, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 3],
        'model__min_samples_leaf': [1]
        }
    )
}

df = pd.read_csv('cost_cars/cars_dataset.csv', sep=';')
cat_cols = ['brand', 'model', 'transmission', 'body_type']
num_cols = df.columns.drop(cat_cols + ['price_rub'])

X = df.drop('price_rub', axis = 1)
Y = df['price_rub']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RS)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
])

preprocessor_scaled  = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe, num_cols)
])

num_pipe_no_scale = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
])

preprocessor_no_scale = ColumnTransformer([
    ('cat', cat_pipe, cat_cols),
    ('num', num_pipe_no_scale, num_cols)
])

car = pd.DataFrame([{
    'brand': 'BMW',
    'model': '5 Series',
    'year': 2020,
    'mileage_km': 55000,
    'engine_volume': 3.0,
    'engine_hp': 300,
    'n_owners': 1,
    'transmission': 'Automatic',
    'body_type': 'Sedan'
}])
real_price = calc_price(car.iloc[0])
print(f'\n Real price = {real_price} RUB')

for name, (model, params) in models.items():
    if name in ['lin_reg', 'ridge']:
        preprocessor = preprocessor_scaled
    else:
        preprocessor = preprocessor_no_scale

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    grid = GridSearchCV(pipe, params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    print('\n' + colored(name.center(20, '-'), 'red'))

    pipe.fit(X_train, y_train)
    baseline_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    r2 = r2_score(y_test, baseline_pred)
    print("BASELINE")
    print(f'RMSE = {round(rmse,0)}')
    print(f'R2 = {round(r2,2)}')    

    if not params:
        predict = pipe.predict(car)[0]
        print(f'Predicted price = {round(predict, 0)} RUB')
        print("No hyperparameters to tune, skipping GridSearchCV")
        continue

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    tuned_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
    r2 = r2_score(y_test, tuned_pred)
    print("TUNED")
    print(f'RMSE = {round(rmse,0)}')
    print(f'R2 = {round(r2,2)}')   
    # print('Best params:',grid.best_params_) 

    predict = best_model.predict(car)[0]
    print(f'{name}: Predicted price = {round(predict, 0)} RUB')




    