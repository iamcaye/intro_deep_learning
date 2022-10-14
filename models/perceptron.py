#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
  def __init__(self, lr=0.01, N=50, state=1):
    self.lr = lr
    self.N = N
    self.state = state

  def fit(self, X, Y):
    rgen = np.random.RandomState(self.state)

    self.W = rgen.nomral(loc=0.0, scale=lr, size=X.shap[1])
    self.B = np.float(0)
    self.errors = []

    for _ in range(self.N):
      err = 0
      for xi, target in zip(X,Y):
        update = self.eta * (target - self.predict(xi))
        self.W += update * xi
        self.B += update
        err += int(update != 0.0)

      self.append(err)

    return self

  def net_input(self, X):
    return np.dot(X, self.W) + self.B

  def predict(self, X):
    return np.where(self.net_input(X) >= 0.0, 1, 0)


if __name__ == "__main__":
  s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
  df = pd.read_csv(s, header=None, encoding='utf-8')
  df.tail()
  # print(df)

  y = np.where(df.iloc[0:100, 4].values == 'Iris-setosa', 0, 1)
  X = df.iloc[0:100, [0,2]].values

  plt.scatter(X[:50, 0], X[:50,1], color='red', marker='o', label='Setosa')
  print(y)

  plt.show()
