"""Small shared helpers for handling the object-typed copula parameter matrices."""

import numpy as np


def _to_scalar(x):
    """Extract a scalar from nested list/array wrappers; return np.nan if not possible."""
    if x is None:
        return np.nan

    # unwrap nested singletons: [x], [[x]], etc.
    while isinstance(x, list) and len(x) == 1:
        x = x[0]

    # unwrap numpy arrays
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return np.nan
        return float(np.ravel(x)[0])

    # plain number
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    return np.nan


def expand_3x3_object_matrix(mat, s):
    """
    mat: object array-like containing nan or scalar-ish entries.
    returns: same shape object np.ndarray where non-nan entries are arrays of shape (s,1).
    """
    mat = np.asarray(mat, dtype=object)
    out = np.empty(mat.shape, dtype=object)

    for idx, val in np.ndenumerate(mat):
        scalar = _to_scalar(val)
        if np.isnan(scalar):
            out[idx] = np.nan
        else:
            out[idx] = np.full((s, 1), scalar)

    return out
