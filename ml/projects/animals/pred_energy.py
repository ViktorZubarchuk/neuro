import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from termcolor import colored
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42)
}

df = pd.read_csv('animals_dataset.csv', sep = ';')

X = df.drop(columns='energy_level', axis = 1)
y = df['energy_level']
X['is_pet_friendly'] = X['is_pet_friendly'].astype('category')
mask = y.notna()
X = X[mask]      
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = X.select_dtypes(exclude=['object','category']).columns
cat_cols = X.select_dtypes(include=['object','category']).columns

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print('\n', colored(name, 'red'))
    print(f'cv_mean = {cv_scores.mean():.2f}')
    print(f'RMSE = {round(rmse, 2)}')
    print(f'MAE = {mae:.2}')
    print(f'R2 = {r2:.2}')
    