import pandas as pd
import fuzzy_system as fs
import numpy as np

df = pd.read_excel("data/example_dataNew_scaled.xlsx"  )
X = df.to_numpy()
print(np.max(X))
# print(X.shape)
# print(np.mean(np.abs(X),axis=0))
# print(np.max(np.abs(X),axis=0))
# print(df.dtypes)
# print(df)


#-Fuzzy-algoruthm--------------------------------------------------------------
space = fs.FuzzyMetric(X)
space.run("test.csv")
