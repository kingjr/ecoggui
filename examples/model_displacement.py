# Authors: Jean-Remi King <jeanremi.king@gmail.com>
#          Nghia Ho <http://nghiaho.com>

"""Generates 10 random 3D coordinates, rotate and translate them, and show how
the displacement can be fitted and predicted with `ModelDisplacement`.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from ecoggui import ModelDisplacement

# Random rotation and translation
R = np.mat(np.random.rand(3, 3))
t = np.mat(np.random.rand(3, 1))

# make R a proper rotation matrix, force orthonormal
U, S, Vt = np.linalg.svd(R)
R = U * Vt

# remove reflection
if np.linalg.det(R) < 0:
    Vt[2, :] *= -1
    R = U * Vt

# number of points
n = 10
X = np.mat(np.random.rand(n, 3))
Y = R * X.T + np.tile(t, (1, n))
Y = Y.T

# To avoid confusion, we'll now treat these matrices as array:
X, Y = np.array(X), np.array(Y)

# recover the transformation
displacer = ModelDisplacement()
displacer.fit(X, Y)
Xt = displacer.transform(X)

# Compute the error between the true Y position and the predicted X position
# from the rotation + translation fit
err = np.sqrt(np.sum((Xt - Y) ** 2))
print(err)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=100, c='k')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], s=100, c='b')
ax.scatter(Xt[:, 0], Xt[:, 1], Xt[:, 2], s=40, c='r')
ax.legend(['X', 'Y', 'predicted X'])
plt.show()
