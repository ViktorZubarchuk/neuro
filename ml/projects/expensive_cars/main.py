import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

param_grid_des_tree = {
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 3, 5],
    'model__criterion': ['gini', 'entropy', 'log_loss']
}

param_grid_rand_forest = {
    'model__max_depth': [3, 5, 7],
    'model__min_samples_split': [2, 3, 5],
    'model__criterion': ['gini', 'entropy', 'log_loss']
}

param_grid_grad_boost = {
    'model__n_estimators': [80, 100, 120],
    'model__learning_rate': [0.1, 0.07, 0.12],
    'model__criterion': ['friedman_mse', 'squared_error']
}

df = pd.read_csv('expensive_cars/cars_dataset.csv', sep=';')

X = df.drop('expensive', axis = 1)
y = df['expensive']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

cat_features = ['brand', 'model', 'transmission', 'body_type']
num_features = [col for col in df.columns if col not in (cat_features + ['expensive'])]

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_features),
    ('num', num_pipeline, num_features)
])

model_des_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('model', DecisionTreeClassifier(random_state=42))
])

model_rand_forest = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

model_grad_boost = Pipeline([
    ('preprocessor', preprocessor),
    ('model', GradientBoostingClassifier(random_state=42))
])

model_des_tree.fit(X_train, y_train)
model_des_tree_y_pred = model_des_tree.predict(X_test)
print("Accuracy model_des_tree:", accuracy_score(y_test, model_des_tree_y_pred))
grid_des_tree = GridSearchCV(model_des_tree, param_grid=param_grid_des_tree, cv=5)
grid_des_tree.fit(X_train, y_train)
grid_des_tree_y_pred = grid_des_tree.predict(X_test)
print("Accuracy grid_des_tree:", accuracy_score(y_test, grid_des_tree_y_pred))

model_rand_forest.fit(X_train, y_train)
model_rand_forest_y_pred = model_rand_forest.predict(X_test)
print("Accuracy model_rand_forest:", accuracy_score(y_test, model_rand_forest_y_pred))
grid_rand_forest = GridSearchCV(model_rand_forest, param_grid=param_grid_rand_forest, cv=5)
grid_rand_forest.fit(X_train, y_train)
grid_rand_forest_y_pred = grid_rand_forest.predict(X_test)
print("Accuracy grid_rand_forest:", accuracy_score(y_test, grid_rand_forest_y_pred))

model_grad_boost.fit(X_train, y_train)
model_grad_boost_y_pred = model_grad_boost.predict(X_test)
print("Accuracy model_grad_boost:", accuracy_score(y_test, model_grad_boost_y_pred))
grid_grad_boost = GridSearchCV(model_grad_boost, param_grid=param_grid_grad_boost, cv=5)
grid_grad_boost.fit(X_train, y_train)
grid_grad_boost_y_pred = grid_grad_boost.predict(X_test)
print("Accuracy grid_grad_boost:", accuracy_score(y_test, grid_grad_boost_y_pred))



