import pandas as pd
import fuzzy_system as fs

df = pd.read_csv("data/data_0.csv")
X = df.to_numpy()[:, 1:]
print(X)
print(X.shape)
# print(df.dtypes)
# print(df)

space = fs.FuzzyMetric(X)
space.run()
