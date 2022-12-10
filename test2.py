import numpy as np
import pandas as pd
# sklearn version: v1.0.1
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, 
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.inspection import (partial_dependence, 
                                PartialDependenceDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Set2')
columns = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare', 
           'adult_male']
df = sns.load_dataset('titanic')[columns].dropna()
X = df.drop(columns='survived')
y = df['survived']
X_train, X_test, y_train, y_test =  train_test_split(
    X, y, random_state=42, test_size=.25
)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
def evaluate(model, X_train, y_train, X_test, y_test):
    name = str(model).split('(')[0]
    print(f"========== {name} ==========")
    y_train_pred = model.predict_proba(X_train)[:,1]
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    print(f"Train ROC AUC: {roc_auc_train:.4f}")
    
    y_test_pred = model.predict_proba(X_test)[:,1]
    roc_auc_test = roc_auc_score(y_test, y_test_pred)
    print(f"Test ROC AUC: {roc_auc_test:.4f}")
    
evaluate(rf, X_train, y_train, X_test, y_test)
var = 'pclass'
PartialDependenceDisplay.from_estimator(rf, X_train, [var]);