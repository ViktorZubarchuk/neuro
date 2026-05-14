import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models_utils import LinearRegression as LinReg

np.random.seed(42)

n = 30
power = np.random.randint(100,300,n)
engine_volume = np.round(np.random.uniform(2.0, 4.0, n), 1)
max_torque = np.random.randint(200, 400, n)
x = np.column_stack([power, engine_volume, max_torque])
w = [200, 5000, 100]
b = 10000
noise = np.random.normal(0, 2000, n)
price = x.dot(w) + b + noise


df = pd.DataFrame({
    'power': power,
    'engine_volume': engine_volume,
    'max_turque': max_torque,
    'price': price
    })

model = LinReg(learning_rate=0.01, epochs=1000)
model.fit(x, price)

pred = model.predict(x)
print(pred[:5])
print(price[:5])
