from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

data_train = pd.read_csv("combined_data_train.csv")
data_celeb = pd.read_csv("combined_data_celebs.csv")
data_celeb["label"] = "baseline"

X_train,y_train = data_train[["h_hitrate", "sentiment", "subjectivity"]], data_train["label"]
y_train = y_train.map({"troll": 1, "baseline": 0})

X_celeb,y_celeb = data_celeb[["h_hitrate", "sentiment", "subjectivity"]], data_celeb["label"]
y_celeb = y_celeb.map({"troll": 1, "baseline": 0})

scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_celeb_transformed = scaler.transform(X_celeb)

rf_clf = RandomForestClassifier(n_estimators=50, max_depth=80, max_features='sqrt')
rf_clf.fit(X_train_transformed, y_train)
rf_y_celeb_pred = rf_clf.predict(X_celeb_transformed)

