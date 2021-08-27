# Overview

This package implements weighted Distance Weighted Discrimination (wDWD). For details see
([Marron et al 2007][marron-et-al], [Wang and Zou 2018][wang-zou]).

The package currently implements:

- Original wDWD formulation solved with Second Order Cone Programming (SOCP) and solved
using cvxpy.

- Genralized DWD (gDWD) and kernel gDWD solved with the Majorization-Minimization
algorithm presented in Wang and Zou, 2018.


Marron, James Stephen, Michael J. Todd, and Jeongyoun Ahn. "Distance-weighted
discrimination." Journal of the American Statistical Association 102, no. 480 (2007):
1267-1271.

Wang, Boxiang, and Hui Zou. "Another look at distance‐weighted discrimination." Journal
of the Royal Statistical Society: Series B (Statistical Methodology) 80, no. 1 (2018):
177-198.

# Installation

The wdwd package can be installed via github. This package is currently only
tested in python 3.6.

[Flit](https://github.com/takluyver/flit) is used for packaging, and all package metadata is stored in `pyproject.toml`. To install this project locally or for development, use `flit install` or build a pip-installable wheel with `flit build`.

# Example

```python
  from sklearn.datasets import make_blobs, make_circles
  from wdwd import WDWD, KernGDWD

  # sample sythetic training data
  X, y = make_blobs(n_samples=200, n_features=2,
                    centers=[[0, 0],
                             [2, 2]])

  # fit wDWD classifier
  wdwd = WDWD(C='auto').fit(X, y)

  # compute training accuracy
  wdwd.score(X, y)

  0.94
```

![dwd_sep_hyperplane][dwd_sep_hyperplane]

```python
# sample some non-linear, toy data
X, y = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)

# fit kernel DWD wit gaussian kernel
kdwd = KernGDWD(lambd=.1, kernel='rbf',
                kernel_kws={'gamma': 1}).fit(X, y)

# compute training accuracy
kdwd.score(X, y)

0.915
```

![kern_dwd][kern_dwd]
