import cvxpy as cp
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.metrics.pairwise import euclidean_distances

from wdwd.utils import pm1
from wdwd.linear_model import LinearClassifierMixin


class WDWD(BaseEstimator, LinearClassifierMixin):

    def __init__(self, C=1.0, solver_kws={}):
        self.C = C
        self.solver_kws = solver_kws

    def fit(self, X, y, pi=[1.0, 1.0], costs=[1.0, 1.0], sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X: {array-like, sparse matrix}, (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y: array-like, (n_samples, )
            Target vector relative to X

        pi: array_like, (2, )
            Class proportions in population. [pi_0, pi_1]

        costs: array-like, (2, )
            Misclassification costs. [cost_fp, cost_fn]
        
        sample_weight : array-like, shape = [n_samples], optional
            Array of weights that are assigned to individual
            samples.

        Returns
        -------
        self : object
        """
        # TODO: what to do about multi-class

        self.classes_ = np.unique(y)

        y_0 = (y == self.classes_[0])
        y_1 = (y == self.classes_[1])
        n_0 = sum(y_0)
        n_1 = sum(y_1)
        n_samples = len(y)
        pi_s = np.array([n_0, n_1]) / n_samples

        pi = np.array(pi)
        costs = np.array(costs)

        if sample_weight is None:
            weights = costs * pi / pi_s
            W = np.ones(n_samples)
            W[y_0] = weights[0]
            W[y_1] = weights[1]
        else:
            W = np.array(sample_weight).reshape(-1)
            len(W) == n_samples
            
        if self.C == 'auto':
            self.C = auto_dwd_C(X, y)

        # fit wighted DWD
        self.coef_, self.intercept_, self.eta_, self.d_, self.problem_ = \
            solve_wdwd_socp(X, y, W, C=self.C, solver_kws=self.solver_kws)

        self.coef_ = self.coef_.reshape(1, -1)
        self.intercept_ = self.intercept_.reshape(-1)

        return self


def solve_wdwd_socp(X, y, W, C=1.0, solver_kws={}):
    """
    Solves distance weighted discrimination optimization problem.

    Parameters
    ----------
    X: (n_samples, n_features)

    y: (n_samples, )

    W: (n_samples, )

    C: float
        Strictly positive tuning parameter.

    solver_kws: dict
        Keyword arguments to cp.solve

    Returns
    ------
    beta: (n_features, )
        DWD normal vector.

    intercept: float
        DWD intercept.

    eta, d: float
        Optimization variables.

    problem: cp.Problem

    """

    if C < 0:
        raise ValueError("Penalty term must be positive; got (C={})".format(C))
        
    X, y = check_X_y(X, y,
                     accept_sparse='csr',
                     dtype='numeric')

    # convert y to +/- 1
    y = pm1(y)

    n_samples, n_features = X.shape

    # problem data
    X = cp.Parameter(shape=X.shape, value=X)
    y = cp.Parameter(shape=y.shape, value=y)
    W = cp.Parameter(shape=W.shape, value=W)
    C = cp.Parameter(value=C, nonneg=True)

    # optimization variables
    beta = cp.Variable(shape=n_features)
    intercept = cp.Variable()
    eta = cp.Variable(shape=n_samples, nonneg=True)

    rho = cp.Variable(shape=n_samples)
    sigma = cp.Variable(shape=n_samples)

    # objective funtion        
    e = np.ones(n_samples)
    objective = e.T @ (rho + sigma + C * cp.multiply(W, eta))

    # setup constraints
    # TODO: do we need explicit SOCP constraints?
    Y_tilde = cp.diag(y)  # TODO: make sparse
    constraints = [cp.multiply(W, rho - sigma) == \
                   Y_tilde @ X @ beta + intercept * y + eta,
                   cp.SOC(cp.Parameter(value=1), beta)]  # ||beta||_2^2 <= 1

    # rho^2 - sigma^2 >= 1
    constraints.extend([cp.SOC(rho[i], cp.vstack([sigma[i], 1]))
                        for i in range(n_samples)])

    # solve problem
    problem = cp.Problem(cp.Minimize(objective),
                         constraints=constraints)

    problem.solve(**solver_kws)

    # d = W * (rho - sigma)
    d = W.value * (rho.value - sigma.value)

    return beta.value, intercept.value, eta.value, d, problem


def auto_dwd_C(X, y, const=100):
    """
    Automatic choice of C from Distance-Weighted Discrimination by Marron et al,
    2007. Note this only is for the SOCP formulation of DWD.

    C = 100 / d ** 2

    Where d is the median distance between points in either class.

    Parameters
    ----------
    X: array-like, (n_samples, n_features)
        The input data.

    y: array-like, (n_samples, )
        The vector of binary class labels.

    const: float
        The constanted used to determine C. Originally suggested to be 100.

    """
    labels = np.unique(y)
    assert len(labels) == 2

    # pariwise distances between points in each class
    D = euclidean_distances(X[y == labels[0], :],
                            X[y == labels[1], :])

    d = np.median(D.ravel())

    return const / d ** 2


def wdwd_obj(X, y, W, C, beta, offset, eta):
    """
    Objective function for DWD.
    """
    d = y * (X.dot(beta) + offset) + eta

    return sum(W / d) + C * sum(eta)
