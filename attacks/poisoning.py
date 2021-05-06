from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def train_and_output_reports(X_train, y_train, X_val, y_val):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_val_transformed = scaler.transform(X_val)
    X_test_transformed = scaler.transform(X_test)

    svm_clf = GridSearchCV(svm.SVC(), svm_grid, refit=True, scoring="precision")
    svm_clf.fit(X_train_transformed, y_train)
    svm_y_val_pred = svm_clf.predict(X_val_transformed)
    svm_val_report = classification_report(y_val, svm_y_val_pred)
    print("svm_val_report:")
    print(svm_val_report)
    print()
    
    rf_clf = GridSearchCV(RandomForestClassifier(), rf_grid, refit=True, scoring='precision')
    rf_clf.fit(X_train_transformed, y_train)
    rf_y_val_pred = rf_clf.predict(X_val_transformed)
    rf_val_report = classification_report(y_val, rf_y_val_pred)
    print("rf_val_report:")
    print(rf_val_report)
    print()
    
    kn_clf = GridSearchCV(KNeighborsClassifier(), kn_grid, refit=True, scoring='precision')
    kn_clf.fit(X_train_transformed, y_train)
    kn_y_val_pred = kn_clf.predict(X_val_transformed)
    kn_val_report = classification_report(y_val, kn_y_val_pred)
    print("kn_val_report:")
    print(kn_val_report)
    print()

def poison(y, p):
    return [s if np.random.rand() < p else (1-s) for s in y]

svm_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1],'kernel': ['rbf', 'poly', 'sigmoid']}
rf_grid = {'n_estimators': [25, 50, 100, 200, 400], 'max_features' : ['auto', 'sqrt'], 'max_depth': [10, 20, 40, 80, None]}
kn_grid = {'n_neighbors': [5, 10, 30, 50, 100, 200], 'weights': ['uniform', 'distance']}

data_train = pd.read_csv("combined_data_train.csv")
data_val = pd.read_csv("combined_data_val.csv")
data_test = pd.read_csv("combined_data_test.csv")

X_train,y_train = data_train[["h_hitrate", "sentiment", "lifespan"]], data_train["label"]
y_train = y_train.map({"troll": 1, "baseline": 0})

X_val,y_val = data_val[["h_hitrate", "sentiment", "lifespan"]], data_val["label"]
y_val = y_val.map({"troll": 1, "baseline": 0})

X_test,y_test = data_test[["h_hitrate", "sentiment", "lifespan"]], data_test["label"]
y_test = y_test.map({"troll": 1, "baseline": 0})

y_train_poisoned_10 = poison(y_train, 0.10)
y_train_poisoned_25 = poison(y_train, 0.25)
y_train_poisoned_50 = poison(y_train, 0.50)
y_val_poisoned_10 = poison(y_val, 0.10)
y_val_poisoned_25 = poison(y_val, 0.25)
y_val_poisoned_50 = poison(y_val, 0.50)

print("Training with 10% poisoned data")
train_and_output_reports(X_train, y_train_poisoned_10, X_val, y_val_poisoned_10)
print()

print("Training with 25% poisoned data")
train_and_output_reports(X_train, y_train_poisoned_25, X_val, y_val_poisoned_25)
print()

print("Training with 50% poisoned data")
train_and_output_reports(X_train, y_train_poisoned_50, X_val, y_val_poisoned_50)
print()
