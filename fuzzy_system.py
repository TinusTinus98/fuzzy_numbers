import numpy as np
from pyearth import Earth
from scipy.stats import spearmanr
import gamma_cor


class FuzzyMetric:
    def __init__(self, X) -> None:
        self.m = X.shape[1]  # number of dimensions
        self.n = X.shape[0]  # number of observations
        self.l = 0
        self.X = X
        self.x_star = np.array([0.5, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.2, 0.3, 0.6])
        self.k = [np.ones((self.m, self.n))]
        self.cfi_list = []
        self.cfi_calculation()

    def cfi_calculation(self):  # Composite fuzzy indicator
        assert len(self.cfi_list) != self.l
        k_matrix = np.array([self.k[self.l] for _ in range(self.n)])
        x_star_matrix = np.array([self.x_star[self.l] for _ in range(self.n)])
        distance_list = np.abs(x_star_matrix - self.X)
        metric_list = np.divide(k_matrix, k_matrix + distance_list)
        cfi = np.prod(metric_list, axis=0)
        self.cfi_list.append(cfi)

    def mars_calculation(self):
        assert len(self.cfi_list) != self.l
        model = Earth()
        model.fit(self.X, self.cfi_calculation[self.l])  # Fit an Earth model
        print(model.trace())  # Print the model
        print(model.summary())

    def correlation(self):
        coef, p = spearmanr(self.cfi_list[self.l], self.cfi_list[self.l - 1])
        print("Spearmans correlation coefficient: %.3f" % coef)
        alpha = 0.05
        if p <= alpha:  # interpret the significance
            print("Samples are correlated (reject H0) p=%.3f" % p)

    def gamma_correlation(self):
        return gamma_cor.generate(self.cfi_list[-1], self.cfi_list[-2])
