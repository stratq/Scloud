import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m

def func(x):
    return 1 / (1 + m.exp(-x))

DD = pd.read_csv("haberman.csv")
print(DD)

x = DD.values[:,:3]
y = DD.values[:,3:]-1

x = np.array(x)
y = np.array(y)

x = np.concatenate([np.ones((x.shape[0],1)), x], axis=1)

res = np.linalg.solve((x.T.dot(x)), x.T.dot(y))
print(res)

W = np.zeros(4)

l = y.shape[0]

n = 0.01

for i in range(200):
    sum=0
    for j in range(4):
      for k in range(l):
          sum += x[k,j] * y[k] *func(-y[k] * W.dot(x[k]))
      W[j] = W[j] + n * (1 / l) * sum

predskazanie = x.dot(W)

for i in range(l):
  sum += 1 if predskazanie[i] == y[i] else -1
sum*=-1
print('Tochnost = ',sum/l)