import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

con = pd.read_excel("data/example_dataNew_scaled.xlsx"  )
cormat = con.corr()
round(cormat,2)
sns.heatmap(cormat)
plt.show()