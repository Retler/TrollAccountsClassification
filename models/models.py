from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#svm_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1],'kernel': ['rbf', 'poly', 'sigmoid']}
svm_grid = {'C': [1], 'gamma': [0.1],'kernel': ['rbf']}
# max_depth: The maximum depth of the tree. If None, then nodes are expanded until
# max_features: How many features to consider at each split
rf_grid = {'n_estimators': [25, 50, 100, 200, 400], 'max_features' : ['auto', 'sqrt'], 'max_depth': [10, 20, 40, 80, None]}
kn_grid = {'n_neighbors': [5, 10, 30, 50, 100, 200], 'weights': ['uniform', 'distance']}

data = pd.read_csv("combined_data.csv")
X,y = data[["h_hitrate", "sentiment", "lifespan"]], data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

svm_clf = GridSearchCV(svm.SVC(), svm_grid, refit=True)
svm_clf.fit(X_train, y_train)

# rf_clf = GridSearchCV(RandomForestRegressor(), rf_grid, refit=True)
# rf_clf.fit(X_train, y_train)

# kn_clf = GridSearchCV(KNeighborsClassifier(), kn_grid, refit=True)
# kn_clf.fit(X_train, y_train)

#grid.predict([[0,0],[0,1],[1,0],[1,1]])
