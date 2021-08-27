# Overview

This package implements weighted Distance Weighted Discrimination (wDWD). For details see
([Marron et al 2007][marron-et-al], [Qiao et al 2010][qiao-et-al], and [Wang and Zou 2018][wang-zou]).

The package currently implements:

- Weighted DWD formulation solved with Second Order Cone Programming (SOCP) and solved
using cvxpy.

- Genralized DWD (gDWD) and kernel gDWD solved with the Majorization-Minimization
algorithm presented in Wang and Zou, 2018.


Marron, James Stephen, Michael J. Todd, and Jeongyoun Ahn. "Distance-weighted
discrimination." Journal of the American Statistical Association 102, no. 480 (2007):
1267-1271.

Qiao, Xingye, Hao Helen Zhang, Yufeng Liu, Michael J. Todd, and J. S. Marron.
"Weighted Distance Weighted Discrimination and Its Asymptotic Properties." 
Journal of the American Statistical Association</i> 105, no. 489 (2010): 401-14.

Wang, Boxiang, and Hui Zou. "Another look at distance‐weighted discrimination." Journal
of the Royal Statistical Society: Series B (Statistical Methodology) 80, no. 1 (2018):
177-198.

# Installation

After cloning the repository, install the package with
```
pip install -e .
```

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

[marron-et-al]: https://amstat.tandfonline.com/doi/abs/10.1198/016214507000001120
[qiao-et-al]: https://www.jstor.org/stable/29747036
[wang-zou]: https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12244

[dwd_sep_hyperplane]: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/dwd_sep_hyperplane.png
[kern_dwd]: https://raw.githubusercontent.com/slicersalt/dwd/master/doc/figures/kern_dwd.png
