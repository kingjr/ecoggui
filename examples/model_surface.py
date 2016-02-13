# Author: Jean-Remi King <jeanremi.king@gmail.com>

"""Generates 3 random 3D coordinates from a known 5x5 grid. Show how the grid
position, rotate and bending can be inferred using `ModelSurface`.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from ecoggui import ModelSurface

# Generate 2D grid of electrodes
n_samples = 6 ** 2
X = np.meshgrid(np.linspace(0, 1., np.sqrt(n_samples)),  # x
                np.linspace(0, 1., np.sqrt(n_samples)))  # y
X = np.mat(np.transpose([ii.ravel() for ii in X]))
X = np.hstack((X, np.zeros((len(X), 1))))  # z

# Slight bending of the surface using polynomial of 2nd order
degree = 2
X_poly = PolynomialFeatures(degree).fit_transform(X)
n_samples, n_coefs = X_poly.shape
coefs = np.zeros((3, n_coefs))
coefs[0, 1] = 1  # x
coefs[1, 2] = 1  # y
coefs[2, 3] = 1  # z
np.random.seed(4)
coefs += np.random.randn(*coefs.shape) ** 3 / 10
X_t = np.dot(coefs, X_poly.T).T

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

Y = np.transpose(np.dot(R, X_t.T) + np.tile(t, (1, n_samples)))
Y = np.array(Y)

# Here alpha controls the constrain on local distance
surface = ModelSurface(alpha=.1)

# Let's say we only know 4 point, can we fit the entire surface
known_idx = np.random.randint(0, n_samples, 4)
surface.fit(X, Y[known_idx], idx=known_idx)
Y_pred = surface.predict(X)

# Plot
fig = plt.figure()
ax = fig.add_subplot(121, aspect='equal')
ax.scatter(X[:, 0], X[:, 1], s=20, color='k')
ax.scatter(X[known_idx, 0], X[known_idx, 1], s=100, color='k')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('known 2D locations')

ax = fig.add_subplot(122, projection='3d', aspect='equal')
for xyz, color in zip((Y, Y_pred), ('r', 'k')):
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=40,
               c=color, color=color, facecolor=color)
    # ax.plot_trisurf(xyz[:, 0], xyz[:, 1], xyz[:, 2],
    #                 edgecolor='r', alpha=.1, linewidth=2.)
ax.scatter(Y[known_idx, 0], Y[known_idx, 1], Y[known_idx, 2], s=100,
           c='k', color='k', facecolor='k')
ax.legend(['true', 'predicted', 'known'])
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_title('known and predicted 3D locations')
plt.show()


# Note the difference between the fit and the truth. This can be explained
# i) because there isn't enough data (change n_known to see how this changes)
# ii) because we impose a distance constrain which may not be respected by the
# polynomial transform, which takes the surface as if it was elastic.
