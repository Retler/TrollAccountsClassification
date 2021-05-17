from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

svm_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1],'kernel': ['rbf', 'poly', 'sigmoid']}
rf_grid = {'n_estimators': [25, 50, 100, 200, 400], 'max_features' : ['auto', 'sqrt'], 'max_depth': [10, 20, 40, 80, None]}
kn_grid = {'n_neighbors': [5, 10, 30, 50, 100, 200], 'weights': ['uniform', 'distance']}

data_train = pd.read_csv("combined_data_train.csv")
data_val = pd.read_csv("combined_data_val.csv")
data_test = pd.read_csv("combined_data_test.csv")

X_train,y_train = data_train[["h_hitrate", "sentiment", "lifespan", "subjectivity"]], data_train["label"]
y_train = y_train.map({"troll": 1, "baseline": 0})

X_val,y_val = data_val[["h_hitrate", "sentiment", "lifespan", "subjectivity"]], data_val["label"]
y_val = y_val.map({"troll": 1, "baseline": 0})

X_test,y_test = data_test[["h_hitrate", "sentiment", "lifespan", "subjectivity"]], data_test["label"]
y_test = y_test.map({"troll": 1, "baseline": 0})

scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_val_transformed = scaler.transform(X_val)
X_test_transformed = scaler.transform(X_test)

svm_clf = GridSearchCV(svm.SVC(), svm_grid, refit=True, scoring="precision", verbose=3)
svm_clf.fit(X_train_transformed, y_train)
svm_y_val_pred = svm_clf.predict(X_val_transformed)
svm_val_report = classification_report(y_val, svm_y_val_pred)

rf_clf = GridSearchCV(RandomForestClassifier(), rf_grid, refit=True, scoring='precision', verbose=3)
rf_clf.fit(X_train_transformed, y_train)
rf_y_val_pred = rf_clf.predict(X_val_transformed)
rf_val_report = classification_report(y_val, rf_y_val_pred)

kn_clf = GridSearchCV(KNeighborsClassifier(), kn_grid, refit=True, scoring='precision', verbose=3)
kn_clf.fit(X_train_transformed, y_train)
kn_y_val_pred = kn_clf.predict(X_val_transformed)
kn_val_report = classification_report(y_val, kn_y_val_pred)

rf_y_test_pred = rf_clf.predict(X_test_transformed)
rf_test_report = classification_report(y_test, rf_y_test_pred)
