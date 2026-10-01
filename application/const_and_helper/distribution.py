
#%%

import numpy as np
import scipy.stats as st
from sklearn.base import BaseEstimator

class OnlineGaussianCopula(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, uniform):

        self.n_observations = uniform.shape[0]
        self.D = uniform.shape[1]

        gauss = st.norm().ppf(uniform)
        self.cov = np.cov(gauss, rowvar=False)
        self.loc = np.mean(gauss, axis=0)

    def _step_update(self, uniform):
        if not uniform.shape[0] == 1:
            raise ValueError("shape should 1 x D")

        self.n_observations += 1
        gauss = st.norm().ppf(uniform).squeeze()
        self.loc = (1 / self.n_observations) * (
            (self.n_observations - 1) * self.loc + gauss
        )
        self.cov = (1 / self.n_observations) * (
            (self.n_observations - 1) * self.cov + np.outer(gauss, gauss)
        )

    def update(self, uniform: np.ndarray):
        if uniform.shape[0] == 1:
            self._step_update(uniform)
        else:
            for i in range(uniform.shape[0]):
                self._step_update(uniform[[i], :])

    def sample(self, shape):
        samples = st.norm().cdf(st.multivariate_normal(self.loc, self.cov).rvs(shape))
        return samples