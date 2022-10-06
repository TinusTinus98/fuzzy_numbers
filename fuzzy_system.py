import numpy as np
from pyearth import Earth


class FuzzyMetric:
    def __init__(self, X) -> None:
        self.m = X.shape[1]  # number of dimensions
        self.n = X.shape[0]  # number of observations
        self.i = 0
        self.X = X
        self.x_star = np.array([0.5, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.2, 0.3, 0.6])
        self.k = [np.ones((self.m, self.n))]
        self.cfi_list = []
        self.cfi_calculation()

    def cfi_calculation(self):  # Composite fuzzy indicator
        assert len(self.cfi_list) != self.i
        k_matrix = np.array([self.k[self.i] for _ in range(self.n)])
        x_star_matrix = np.array([self.x_star[self.i] for _ in range(self.n)])
        distance_list = np.abs(x_star_matrix - self.X)
        metric_list = np.divide(k_matrix, k_matrix + distance_list)
        cfi = np.prod(metric_list, axis=0)
        self.cfi_list.append(cfi)

    def mars_calculation(self):
        assert len(self.cfi_list) != self.i
        model = Earth()
        model.fit(self.X, self.cfi_calculation[self.i])  # Fit an Earth model
        print(model.trace())  # Print the model
        print(model.summary())
