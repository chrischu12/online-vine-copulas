"""Shared data-generating helpers for the vine-copula simulation studies.

These build the covariates, (optionally time-varying) coefficients and the true
copula-family matrix used by the Monte Carlo scripts in ``simulations/``. All
randomness flows through an explicit ``numpy`` generator seeded per call, so the
helpers are reproducible and safe to call inside parallel workers.
"""

import numpy as np


def make_X(N, p=20, seed=None):
    """Design matrix with the first six columns binary and the rest standard normal."""
    rng = np.random.default_rng(seed)

    X = rng.normal(size=(N, p))
    X[:, :6] = rng.binomial(1, 0.5, size=(N, 6))

    return X


def make_beta_t(N, beta, break_points=None, break_indices=None, break_values=None):
    """Repeat ``beta`` over ``N`` steps, optionally applying structural breaks."""
    beta = np.asarray(beta).reshape(-1, 1)
    beta_t = np.repeat(beta[None, :, :], N, axis=0)

    if break_points is not None:
        for bp, idx, val in zip(break_points, break_indices, break_values):
            beta_t[bp:, idx, 0] = val

    return beta_t


def generate_true_copula_matrix(a, cops, seed=None):
    """Draw a random true copula-family matrix on the lower triangle of the D-vine."""
    rng = np.random.default_rng(seed)
    dim = a.shape[0]

    C_true_d = np.full((dim, dim), np.nan, dtype=float)
    for i in range(dim - 1):
        for j in range(dim - 1 - i):
            C_true_d[i, j] = rng.choice(cops)

    return C_true_d
