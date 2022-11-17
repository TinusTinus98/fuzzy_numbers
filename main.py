import pandas as pd
import fuzzy_system as fs
import numpy as np
np.warnings.filterwarnings('ignore')
import sys

df = pd.read_csv("data/data_1.csv")
X = df.to_numpy()[:, :-1]
print(X)
print(X.shape)
print(np.mean(np.abs(X),axis=0))
print(np.max(np.abs(X),axis=0))
# print(df.dtypes)
# print(df)

space = fs.FuzzyMetric(X)
space.run("test.csv")
