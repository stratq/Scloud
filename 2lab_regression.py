import numpy as np
import numpy.linalg as li
import scipy.integrate as itg
import sympy as sp
import math as m
import matplotlib.pyplot as plt
import pandas as pd

DD = pd.read_csv("haberman.csv")
print(DD)

X = DD.values[:,:3]
Y = DD.values[:,3:]-1

X_mean = X.mean()
X2_mean = (X * X).mean()
Y_mean = Y.mean()
XY_mean = (X * Y).mean()

denominator = X2_mean - X_mean**2
a = (XY_mean - X_mean * Y_mean) / denominator
b = (Y_mean * X2_mean - X_mean * XY_mean) / denominator

Y_predict = a*X+b
print(Y_predict)

plt.scatter(X, Y, color="red")
plt.plot(X, Y_predict, color="blue")
plt.show()