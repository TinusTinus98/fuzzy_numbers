from sklearn.inspection import partial_dependence
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

X, y = make_friedman1()
est1 = LinearRegression().fit(X, y)
est2 = RandomForestRegressor().fit(X, y)
value = partial_dependence(est1, X, [1], kind="average")
print(value)
