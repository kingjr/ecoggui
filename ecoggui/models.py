# Author: Jean-Remi King <jeanremi.king@gmail.com>

import numpy as np
from scipy import optimize
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import PolynomialFeatures


class ModelDisplacement(object):
    """Transformer to fit rigid object rotation + translation. It 1) centers
    the data, 2) rotates it with SVD, and 3) fits translation to solve:
    R * X + t = Y
    """
    # Adapted from Nghia Ho <http://nghiaho.com/?page_id=671>
    def fit(self, X, Y):
        """
        Parameters
        ==========
        X : np.matrix, shape(n_points, 2)
            Points 2D locations.
        Y : np.matrix, shape(n_points, 3)
            Points 3D locations.

        Attributes
        ==========
        R_ : np.matrix, shape(3, 3)
            Rotation matrix of f(X, Y) solved with SVD.
        t_ : np.matrix, shape(1, 3)
            Translation vector of f(X, Y) solved after SVD.
        inv_R_ : np.matrix, shape(3, 3)
            Rotation matrix of f(Y, X) solved with SVD.
        t_ : np.matrix, shape(1, 3)
            Translation vector of f(Y, X) solved after SVD.
        """
        self.R_, self.t_ = self._fit(X, Y)
        self.inv_R_, self.inv_t_ = self._fit(Y, X)  # XXX should be analytical!

    def _fit(self, X, Y):
        from numpy import mean, tile, transpose, linalg
        X, Y = np.mat(X), np.mat(Y)
        n_samples = X.shape[0]  # total points

        # Center
        centroid_X = mean(X, axis=0)
        centroid_Y = mean(Y, axis=0)
        XX = X - tile(centroid_X, (n_samples, 1))
        YY = Y - tile(centroid_Y, (n_samples, 1))

        # Fit rotation through SVD
        H = transpose(XX) * YY
        U, S, Vt = linalg.svd(H)
        R_ = Vt.T * U.T

        # Special reflection case
        if linalg.det(R_) < 0:
            Vt[2, :] *= -1
            R_ = Vt.T * U.T

        # Fit translation
        t_ = -R_ * centroid_X.T + centroid_Y.T
        return R_, t_

    def transform(self, X):
        """
        Parameters
        ==========
        X : np.matrix, shape(n_points, 2)
            All points 2D locations.

        Returns
        =======
        Y: np.array, shape(n_points, 3)
            Rotated + translated X
        """
        X = np.mat(X)
        n_samples = X.shape[0]
        y_pred = self.R_ * X.T + np.tile(self.t_, (1, n_samples))
        return np.array(y_pred.T)

    def inverse_transform(self, X):
        X = np.mat(X)
        n_samples = X.shape[0]
        y_pred = self.inv_R_ * X.T + np.tile(self.inv_t_, (1, n_samples))
        return np.array(y_pred.T)

    def fit_transform(self, X, Y):
        """ Fits R*X + t = Y and return predicted X.

        Parameters
        ==========
        X : np.matrix, shape(n_points, 2)
            All points 2D locations.

        Returns
        =======
        Y: np.array, shape(n_points, 3)
            Rotated + translated X
        """
        self.fit(X, Y)
        return np.array(self.t_ransform(X))


class ModelSurface(object):
    """Fits a surface for which we know:
        - a small set (`idx`) of points 3D locations (`y`)
        - all 2D distances of the points (`X`)
    by successively using a rotation fit, translation fit, polynomial fit with
    a constrain on the points close distances.

    Parameters
    =========
    alpha : float, in [0, 1]
        Constrain parameter. If alpha = 1 the polynomial is fully constrained
        by the known 2D distances (rigid surface). If 0, the 2D distance is
        relaxed (more curvature).
    degree : int
        Degree of the polynomial fit.
    """
    def __init__(self, alpha=.5, degree=2, verbose='debug'):
        self.degree = degree
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, X, y, idx):
        """Fit a
        Parameters
        ==========
        X : np.matrix, shape(n_points, 2)
            All points 2D locations.
        Y : np.matrix, shape(n_known, 3)
            Subselection of known points 3D locations.
        idx : list | np.array, shape(n_known,)
            Indices indicating which are the known points.

        Attributes
        ==========
        R_ : np.matrix, shape(3, 3)
            Rotation matrix solved with SVD.
        t_ : np.matrix, shape(1, 3)
            Translation vector solved after SVD.
        coefs_ : np.matrix, shape(3, n_polynomes)
            Polynomial coefficients.
        """

        X = self._check_input(X)
        y = self._check_input(y)
        # Fit Translation and rotation
        self._displacer = ModelDisplacement()
        self._displacer.fit(X[idx], y)

        # It's computationally easier to rotate y, so that we only computes
        # the X polynomial once.
        y_displaced = self._displacer.inverse_transform(y)

        # Compute X polynomial bases
        X_poly = PolynomialFeatures(self.degree).fit_transform(X)
        n_coefs = X_poly.shape[1]

        # Compute distance and neighbors on the 2D grid once
        dist_2D = squareform(pdist(X))
        # normalize distance constrain by distance, add one to avoid / 0
        weights = 1 + dist_2D ** 2

        # Let's initialize from a flat grid since it's the most likely
        # For this we'll put x=1, y=1, z=1
        x0 = np.hstack((np.zeros((3, 1)),
                        np.identity(3),
                        np.zeros((3, n_coefs - 4)))).ravel()
        # Avoid 0. for numerical issues
        x0 += np.random.randn(*x0.shape) / 1000
        coefs_ = optimize.fmin_bfgs(self._loss, x0=x0, gtol=.01,
                                    args=(X_poly, y_displaced, idx, dist_2D,
                                          self.alpha, weights))
        self.coefs_ = coefs_

    @property
    def R_(self):
        return self._displacer.R_

    @property
    def t_(self):
        return self._displacer.t_

    def _check_input(self, X):
        """Transforms 2D to 3D"""
        X = np.array(X)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] == 2:
            X = np.hstack((X, np.zeros((len(X), 1))))
        return X

    def predict(self, X):
        """Predicts the 3D location of from the 2D location.

        Parameters
        ==========
        X : np.matrix, shape(n_points, 2)
            Known 2D locations.

        Returns
        =======
        Y: np.array, shape(n_points, 3)
            Predicted 3D locations.
        """
        X = self._check_input(X)

        # compute polynomial
        X_poly = PolynomialFeatures(self.degree).fit_transform(X)

        # Combine polynomials
        coefs = self.coefs_.reshape([3, -1])
        y_pred = np.dot(coefs, X_poly.T).T

        # Rotate and translate
        return self._displacer.transform(np.array(y_pred))

    def _loss(self, coefs, X_poly, y_displaced, idx, dist_2D, alpha, weights):
        """Least Square Regression on polynomial bases constained with
        known 2D distances"""
        coefs = coefs.reshape([3, -1])
        y_pred = np.dot(coefs, X_poly.T).T

        # Compute error with known points
        y_error = np.linalg.norm(y_pred[idx, :] - y_displaced)

        # Compute points distance on polynomial
        dist_3D = squareform(pdist(y_pred))

        # Compute distance difference weighted by the distance
        dist = np.sqrt(np.sum((dist_3D - dist_2D) ** 2 / weights))
        if self.verbose == 'debug':
            print(y_error, dist)
        return (1 - alpha) * y_error + alpha * dist
