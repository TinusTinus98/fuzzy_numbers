import numpy as np
from tqdm import tqdm
from time import sleep
from scipy.stats import kendalltau, spearmanr
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
        self.p = 10
        # self.s_permutation=[[i for i in range(self.m)]]
        self.x_star = np.array([1.83 for _ in range(10)])
        self.val_0 = 0.5
        self.k = [np.ones(self.m) * self.val_0]
        self.cfi_list = []
        self.cfi_calculation()
        self.mse = []
        self.mars_model = None
        self.gamma = []
        self.spearman = []
        self.kendall = []
        self.limit = 1
        self.limit_p = 0.05
        self.corr_choice = corr_choice
        self.corr = 10
        self.p_value = 10

    def run(self, outfile: str):
        """This function run the Fuzzy algorithm

        Args:
            outfile (str): path of the file you want to store the results. Results are a cfi list, k indicator lists and correlations with p-values
        """
        corr_test = self.corr < self.limit and self.p_value < self.limit_p
        for _ in tqdm(range(self.p), desc="Fuzzy system : "):
            sleep(0.1)
            self.cfi_calculation()
            self.mars_prediction()
            self.indicators_calculation()
            self.correlation()
            self.gamma_correlation()
            self.l += 1
            print("turn : ", self.l)
            corr_test = self.corr < self.limit and self.p_value < self.limit_p
            self.save(outfile)
            if corr_test:
                return None

    def corr_choice(self):
        if self.corr_choice == "kendall":
            self.corr = self.kendall[self.l][0]
            self.p_value = self.kendall[self.l][1]

    def save(self, out_file: str):
        """save on a csv file the data the cfi lists, k indicators list and correlation parameters

        Args:
            out_file (str): path of the csv file you want to store your data
        """
        out = []
        for l in range(self.l):
            dico = {
                "l": self.l,
                "k_list": self.k[l],
                "cfi_list": self.cfi_list[l],
                "mse": self.mse[l],
                "kendall_tau": self.kendall[l][0],
                "kendall_tau_p": self.kendall[l][1],
                "spearman": self.spearman[l][0],
                "spearman_p": self.spearman[l][1],
                "gamma": self.gamma[l],
            }
            out.append(dico)
        df = pd.DataFrame(out)
        df.to_csv(out_file)

    def cfi_calculation(self):
        """compute the cfi indicator with the k indicators using the t-norm. the new cfi list is added to self.cfi_list
        """
        assert len(self.cfi_list) != self.l - 1
        k_matrix = np.array([self.k[self.l] for _ in range(self.n)])
        x_star_matrix = np.array([self.x_star for _ in range(self.n)])
        distance_list = np.abs(x_star_matrix - self.X)
        metric_list = np.divide(k_matrix, k_matrix + distance_list)
        cfi = np.prod(metric_list, axis=1)
        self.cfi_list.append(cfi)

    def mars_prediction(self):
        assert len(self.cfi_list) != self.l
        mse, model = mars.mars_calculation(self.X, self.cfi_list[-1], self.cfi_list)
        self.mars_model = model
        self.mse.append(mse)
        print(model.trace())  # Print the model
        print(model.summary())

    def correlation(self):
        x1 = self.cfi_list[self.l]
        x2 = self.cfi_list[self.l - 1]
        coef, p = spearmanr(x1, x2)
        self.spearman.append([coef, p])
        tau, p_value = kendalltau(x1, x2)
        self.kendall.append([tau, p_value])

    def gamma_correlation(self):
        self.gamma.append(gamma_cor.generate(self.cfi_list[-1], self.cfi_list[-2]))

    def indicators_calculation(self):
        out = []
        for j in range(self.m):
            # x_s = np.array([self.X for _ in range(self.n)])
            array_f_s = np.array([0.0 for _ in range(self.n)])
            for i in range(self.n):
                x_s = np.copy(self.X)
                x_s[:, j] = [self.X[i][j] for _ in range(self.n)]
                f_hat_s = self.mars_model.predict(x_s)
                array_f_s[i] = np.sum(f_hat_s) / self.n
            mean_f_s = np.sum(array_f_s) / self.n
            value = np.sqrt(np.sum(np.square(array_f_s - mean_f_s)) / (self.n - 1))
            out.append(value)
        self.k.append(np.array(out))
