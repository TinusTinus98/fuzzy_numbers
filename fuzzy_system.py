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
        self.i = []
        # self.s_permutation=[[i for i in range(self.m)]]
        self.x_star = np.array([0.5, 0.4, 0.5, 0.6, 0.7, 0.4, 0.5, 0.2, 0.3, 0.6])
        self.k = [np.ones((self.m, self.n))]
        self.cfi_list = []
        self.cfi_calculation()
        self.mars_model = None

    def cfi_calculation(self):  # Composite fuzzy indicator
        assert len(self.cfi_list) != self.l
        k_matrix = np.array([self.k[self.l] for _ in range(self.n)])
        x_star_matrix = np.array([self.x_star[self.l] for _ in range(self.n)])
        distance_list = np.abs(x_star_matrix - self.X)
        metric_list = np.divide(k_matrix, k_matrix + distance_list)
        cfi = np.prod(metric_list, axis=0)
        self.cfi_list.append(cfi)

    def s_calculation(self):
        s = 0
        self.s.append(s)

    def mars_prediction(self):
        assert len(self.cfi_list) != self.l
        model = Earth()
        model.fit(self.X, self.cfi_calculation[self.l])  # Fit an Earth model
        self.mars_model = model
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

    def indicators_calculation(self):
        out = []
        for j in range(self.m):
            x_s = np.array([self.X for _ in range(self.n)])
            array_f_s = np.array([0.0 for _ in range(self.n)])
            for i in range(self.n):
                x_s = np.copy(self.X)
                x_s[:, j] = [self.X[i][j] for _ in range(self.n)]
                self.mars_model.predict()
                f_hat_s = np.array([self.mars_model.predict(x) for x in x_s])
                f_s = np.sum(f_hat_s) / self.n
                array_f_s[i] = f_s
            sum_f_s = np.sum(array_f_s) / self.n
            value = np.sqrt(np.sum(np.square(array_f_s - sum_f_s)) / (self.n - 1))
            out.append(value)
        self.i.append(out)
