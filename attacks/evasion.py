from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

data_train = pd.read_csv("combined_data_train.csv")
data_test = pd.read_csv("combined_data_evasion_test.csv")
X_train, y_train =  data_train[["h_hitrate", "sentiment", "lifespan"]], data_train["label"]
y_train = y_train.map({"troll": 1, "baseline": 0})
X_test, y_test =  data_test[["h_hitrate", "sentiment", "lifespan"]], data_test["label"]
y_test = y_test.map({"troll": 1, "baseline": 0})

scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

rf_clf = RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=80)
rf_clf.fit(X_train_transformed, y_train)
rf_y_test_pred = rf_clf.predict(X_test_transformed)
rf_test_report = classification_report(y_test, rf_y_test_pred)
print(rf_test_report)
