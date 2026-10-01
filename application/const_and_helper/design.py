"""Shared design-matrix builder for the data-preparation scripts.

Both ``01_prepare_data_macro_marginals.py`` and ``01_prepare_data_macro_copula.py``
build design matrices block by block. This replaces the duplicated module-global
``reset_design`` / ``add_block`` / ``build_X_df`` pattern with a small stateful
builder, so each region gets its own instance (no shared mutable globals).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer, OneHotEncoder


class DesignBuilder:
    """Accumulate covariate blocks and assemble them into a named DataFrame.

    Usage:
        b = DesignBuilder()
        b.add_block("t", t_column)
        b.spline_basis(df, "doy", 20, "spline(doy)")
        X_df = b.to_df()
        blocks = b.blocks          # block metadata (for prediction / inspection)
    """

    def __init__(self):
        self.blocks = {}   # name -> {"start", "end", "transformer"}
        self.X_list = []   # list of np arrays (blocks)
        self.idx = 0       # running column index

    def add_block(self, name, M, transformer=None):
        """Append a block of columns and record its span + optional transformer."""
        start = self.idx
        end = self.idx + M.shape[1]
        self.idx = end
        self.X_list.append(M)
        self.blocks[name] = {"start": start, "end": end, "transformer": transformer}
        return M

    def spline_basis(self, df, colname, n_knots, name):
        """Cubic B-spline basis, fit on the training rows only (no leakage)."""
        spl = SplineTransformer(
            degree=3,
            n_knots=n_knots,
            include_bias=False,
            extrapolation="linear",
            knots="quantile",
        )
        spl.fit(df.loc[df["flag"].eq("train"), [colname]])
        basis_vectors = spl.transform(df.loc[:, [colname]])
        return self.add_block(name, basis_vectors, spl)

    def factor_basis(self, df, colname, name):
        """One-hot encode a categorical column (drop first to avoid collinearity)."""
        enc = OneHotEncoder(drop="first", sparse_output=False)
        B = enc.fit_transform(df[[colname]])
        return self.add_block(name, B, enc)

    def factor_smooth_interaction(self, B_spline, B_factor, name):
        """Row-wise products of a spline basis with each factor dummy."""
        parts = [B_spline * B_factor[:, [k]] for k in range(B_factor.shape[1])]
        return self.add_block(name, np.hstack(parts))

    def to_df(self):
        """Assemble the accumulated blocks into a DataFrame with generated names."""
        X = np.hstack(self.X_list)
        colnames = []
        for name, meta in self.blocks.items():
            width = meta["end"] - meta["start"]
            if width == 1:
                colnames.append(name)
            else:
                colnames.extend([f"{name}_{i}" for i in range(width)])
        return pd.DataFrame(X, columns=colnames)
