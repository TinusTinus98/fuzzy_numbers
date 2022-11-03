import numpy as np

# from pyearth import Earth
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import gamma_cor
import mars
import pandas as pd


class FuzzyMetric:
    def __init__(self, X, corr_choice="kendall") -> None:
        self.m = X.shape[1]  # number of dimensions
        self.n = X.shape[0]  # number of observations
        self.l = 0
        self.X = X
        self.i = []
        self.p = 100
        # self.s_permutation=[[i for i in range(self.m)]]
        self.x_star = np.array([5.0 for _ in range(10)])
        self.k = [np.ones(self.m) * 0.5]
        self.cfi_list = []
        self.cfi_calculation()
        self.mars_model = None
        self.gamma = []
        self.spearman = []
        self.kendall = []
        self.limit = 1
        self.limit_p = 0.05
        self.corr_choice = corr_choice
        self.corr = 10
        self.p_value = 10

    def run(self):
        corr_test = self.corr < self.limit and self.p_value < self.limit_p
        while self.l < self.p or corr_test:
            self.cfi_calculation()
            self.mars_prediction()
            self.indicators_calculation()
            self.correlation()
            self.gamma_correlation()
            self.l += 1

    def corr_choice(self):
        if self.corr_choice == "kendall":
            self.corr = self.kendall[self.l][0]
            self.p_value = self.kendall[self.l][1]

    def save(self, out_file):
        out = []
        for l in range(self.l):
            dico = {
                "l": self.l,
                "k_list": self.k[l],
                "cfi_list": self.cfi_list[l],
                "kendall_tau": self.kendall[l][0],
                "kendall_tau_p": self.kendall[l][1],
                "spearman": self.spearman[l][0],
                "spearman_p": self.spearman[l][1],
                "gamma": self.gamma[l],
            }
        df = pd.DataFrame(out)
        df.to_csv(out_file)

    def cfi_calculation(self):  # Composite fuzzy indicator
        assert len(self.cfi_list) != self.l - 1
        k_matrix = np.array([self.k[self.l] for _ in range(self.n)])
        x_star_matrix = np.array([self.x_star for _ in range(self.n)])
        distance_list = np.abs(x_star_matrix - self.X)
        metric_list = np.divide(k_matrix, k_matrix + distance_list)
        cfi = np.prod(metric_list, axis=0)
        self.cfi_list.append(cfi)

    def mars_prediction(self):
        assert len(self.cfi_list) != self.l
        mse, model = mars.mars_calculation(self.X, self.cfi_list[-1], self.cfi_list)
        self.mars_model = model
        # print(model.trace())  # Print the model
        # print(model.summary())

    def correlation(self):
        x1 = self.cfi_list[self.l]
        x2 = self.cfi_list[self.l - 1]
        coef, p = spearmanr(x1, x2)
        # print("Spearmans correlation coefficient: %.3f" % coef)
        alpha = 0.05
        # if p <= alpha:  # interpret the significance
        #     print("Samples are correlated (reject H0) p=%.3f" % p)
        self.spearman.append([coef, p])
        tau, p_value = kendalltau(x1, x2)
        self.kendall.append([tau, p_value])

    def gamma_correlation(self):
        self.gamma.append[gamma_cor.generate(self.cfi_list[-1], self.cfi_list[-2])]

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
        self.k.append(np.array(out))
