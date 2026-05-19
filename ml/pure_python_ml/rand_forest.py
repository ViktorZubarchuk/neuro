from models_utils import RandomForest
import numpy as np

# рост (см), вес (кг)
X_train = np.array([
    [160, 55],
    [165, 60],
    [170, 65],
    [175, 70],
    [180, 85],
    [185, 90],
    [170, 95],
    [160, 80]
])

# 0 = не полный
# 1 = полный
y_train = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
])

X_test = np.array([
    [170, 60],   # не полный
    [175, 95]    # полный
])

model = RandomForest(5)
model.fit(X_train, y_train)

pred = model.predict(X_train)

print("pred:", pred)
print("true:", y_train)
print(model.predict(X_test))
