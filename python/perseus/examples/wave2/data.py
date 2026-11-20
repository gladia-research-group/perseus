"""Small standard datasets (UCI breast cancer / iris / wine, as bundled with scikit-learn) without a download:
scikit-learn's loaders when it is installed, else its CSVs from $PERSEUS_DATA_DIR (the files breast_cancer.csv,
iris.csv, wine_data.csv: first line "n_samples,n_features,class names...", then rows of features and the class);
`blobs` is synthetic (Gaussian clusters)."""
from __future__ import annotations

import os

import numpy as np

FILES = {"breast_cancer": "breast_cancer.csv", "iris": "iris.csv", "wine": "wine_data.csv"}


def load(name: str):
    """(X, y) float64 features and int labels."""
    try:
        from sklearn import datasets
        d = getattr(datasets, f"load_{name}")()
        return np.asarray(d.data, dtype=np.float64), np.asarray(d.target, dtype=np.int64)
    except ImportError:
        pass
    root = os.environ.get("PERSEUS_DATA_DIR")
    if not root:
        raise FileNotFoundError(f"dataset {name}: install scikit-learn or set PERSEUS_DATA_DIR to its CSVs")
    with open(os.path.join(root, FILES[name])) as f:
        n, d = (int(v) for v in f.readline().split(",")[:2])
        a = np.loadtxt(f, delimiter=",")
    assert a.shape == (n, d + 1), (a.shape, n, d)
    return a[:, :d], a[:, d].astype(np.int64)


def available(name: str) -> bool:
    try:
        load(name)
        return True
    except (FileNotFoundError, OSError):
        return False


def scaled(X, lo=-1.0, hi=1.0, ref=None):
    """Min-max scaling of the features to [lo, hi] (on `ref`'s range when given: the training split's)."""
    R = X if ref is None else ref
    mn, mx = R.min(0), R.max(0)
    return lo + (hi - lo) * (X - mn) / np.where(mx > mn, mx - mn, 1.0)


def blobs(rng, N, K, d, spread=0.15, box=0.6):
    """N points in K Gaussian clusters (centres uniform in [-box, box]^d, std `spread`), clipped to [-1, 1]."""
    C = rng.uniform(-box, box, (K, d))
    y = np.arange(N) % K
    X = np.clip(C[y] + spread * rng.standard_normal((N, d)), -1.0, 1.0)
    return X, y, C
