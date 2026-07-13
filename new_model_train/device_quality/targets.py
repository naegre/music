import numpy as np
from scipy.stats import rankdata


class PercentileTargetTransformer:
    def fit(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            raise ValueError("Cannot fit percentile target transformer on no values")
        self.sorted_ = np.sort(values)
        return self

    def fit_transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.fit(values)
        ranks = rankdata(values, method="average")
        return 100.0 * (ranks - 0.5) / len(values)

    def transform(self, values):
        values = np.asarray(values, dtype=np.float64)
        left = np.searchsorted(self.sorted_, values, side="left")
        right = np.searchsorted(self.sorted_, values, side="right")
        midpoint = 0.5 * (left + right)
        return np.clip(100.0 * midpoint / len(self.sorted_), 0.0, 100.0)
