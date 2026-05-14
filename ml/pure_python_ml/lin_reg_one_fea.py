import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models_utils import LinearRegressionOneFeature as LrOneFea
np.random.seed(42)

true_w = 4
true_b = 3
x = np.random.rand(100)
y = x * true_w + true_b + np.random.rand(100)

model = LrOneFea(learning_rate=0.01, epochs=2000)
model.fit(x, y)

y_pred = model.predict(x)

plt.scatter(x, y, label = 'Истинные метки', color = 'blue')   
plt.plot(x, y_pred, label = 'Предсказания', color ='red')
plt.show()