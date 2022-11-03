# https://github.com/scikit-learn-contrib/py-earth
from pyearth import Earth
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import make_regression


def mars_calculation(X, y, cfi):
    model = Earth()
    model.fit(X, y)  # Fit an Earth model
    mse = cross_validation(model, X, y, cfi, n_splits=10, n_repeats=10)
    print(model.trace())  # Print the model
    print(model.summary())
    return mse, model


def cross_validation(model, X, y, cfi, n_splits=10, n_repeats=10):
    # specify cross-validation method to use to evaluate model
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1)
    # evaluate model performance
    scoring = "neg_mean_absolute_error"
    x_cfi = np.hstack((X, np.array([[x] for x in cfi])))
    scores = cross_val_score(model, X, y, scoring=scoring, cv=cv, n_jobs=-1)
    return np.mean(scores)  # print results
