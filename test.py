import numpy as np
from util import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from math import *

my_matrix = np.ndarray(3)
my_matrix[0] = 1
my_matrix[1] = 2
my_matrix[2] = 3
create_first_matrix(my_matrix)

b = my_new_array()
print(b.shape)

a1 = np.ndarray((2, 2))
a2 = np.ndarray((4, 4))
a3 = np.ndarray((5, 2))
check_random_matrix(a1, a2, a3)

a4 = np.dot(a3, a1)

X, y = make_blobs(n_samples=100, n_features=2, centers=2, random_state=0)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)

plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.show()

def initialisation(X):
    W = np.array([[X], [X]])
    b = np.array([X])
    return (W, b)

def model(X, W, b):
    Z = X * W + b
    A = 1 / np.exp(1+e-Z)
    return A

print(model(5, 6, 7))