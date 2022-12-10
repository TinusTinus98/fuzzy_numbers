import pandas as pd
import fuzzy_system as fs
import numpy as np
import plot_tools as pt

df = pd.read_csv("test.csv"  )
print(df.head())
print(df.dtypes)
print(df["cfi_list"])
# pt.simple_plot(Y,X)