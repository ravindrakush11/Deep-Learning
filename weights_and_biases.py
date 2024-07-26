import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
np.random.seed(0)
num_samples = 100
X = np.random.randn(num_samples, 2)
y = (X[:,0]+ X[:,1]>0).astype(int).reshape(num_samples, 1)

# Initialize weights and biases
input_dim = 2
hidden_dim = 4
output_dim = 1

W1 = np.random.rand(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

# Activation function
def relu(x):
  return np.maximum(0, x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def forward_propagation(X):
  Z1 = np.dot(X, W1) + b1
  A1 = relu(Z1)
  Z2 = np.dot(A1, W2) + b2
  A2 = sigmoid(Z2)
  return Z1, A1, Z2, A2

def compute_loss(y_true, y_pred):
  m = y_true.shape[0]
  loss = -1/m * (np.dot(y_true.T, np.log(y_pred)) + np.dot((1 - y_true).T, np.log(1 - y_pred)))
  return np.squeeze(loss)

def backward_propagation(X, y, Z1, A1, Z2, A2):
  m = y.shape[0]
  dZ2 = A2 - y
  dW2 = 1/m * np.dot(A1.T, dZ2)
  db2 = 1/m * np.sum(dZ2, axis=0, keepdims=True)
  dA1 = np.dot(dZ2, W2.T)
  dZ1 = dA1 * (Z1 > 0)
  dW1 = 1/m * np.dot(X.T, dZ1)
  db1 = 1/m * np.sum(dZ1, axis=0, keepdims=True)
  return dW1, db1, dW2, db2

# Training the network
epochs = 1000
learning_rate = 0.01
losses = []

for epoch in range(epochs):
  Z1, A1, Z2, A2 = forward_propagation(X)
  loss = compute_loss(y, A2)
  losses.append(loss)
  dW1, db1, dW2, db2 = backward_propagation(X, y, Z1, A1, Z2, A2)
  W1 -=learning_rate * dW1
  b1 -= learning_rate * db1
  W2 -= learning_rate * dW2
  b2 -= learning_rate * db2

  if epoch % 100 == 0:
    print(f'Epoch {epoch}, Loss: {loss}')

# Model Evaluation
def predict(X):
  _,_,_, A2 = forward_propagation(X)
  return (A2 > 0.5).astype(int)

y_pred = predict(X)
accuracy = np.mean(y_pred==y)
print(f'Accuracy: {accuracy: .4f}')

# Plot graph
def plot_decision_boundary(X, y, predict_func):
  x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
  y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1

  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  grid = np.c_[xx.ravel(), yy.ravel()]
  Z = predict_func(grid)
  Z = (Z > 0.5).astype(int)
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o')
  plt.show()

plot_decision_boundary(X, y, predict)
