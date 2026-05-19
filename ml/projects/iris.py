from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'model__n_neighbors':[3, 5, 7, 9],
    'model__leaf_size': [10, 20, 30, 40]
}

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier()),
])

grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print(grid.best_params_)
print(grid.best_score_)
print(accuracy_score(y_test, y_pred))

