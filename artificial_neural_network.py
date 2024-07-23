# !pip install tensorflow keras

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Synthetic data
np.random.seed(0)
num_sample = 100
X = np.random.randn(num_sample, 2)
y = (X[:,0] + X[:, 1] > 0).astype(int)

# Model building
model = Sequential([
    Dense(4, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])
# Model compiling
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Model training
model.fit(X, y, epochs=67, batch_size=10)

# Model evaluation
loss, accuracy = model.evaluate(X, y)
print(f'Accuracy: {accuracy: .4f}')

# Visualization
import matplotlib.pyplot as plt
def plot_decision_boundary(X, y, model):
  x_min, x_max = X[:, 0].min() - 1, X[:,0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:,1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  grid = np.c_[xx.ravel(), yy.ravel()]
  Z = model.predict(grid)
  Z = (Z > 0.5).astype(int).reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
  plt.show()
plot_decision_boundary(X, y, model)
