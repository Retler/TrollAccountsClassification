from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def poison(y, p):
    return [(1-s) if np.random.rand() < p else s for s in y]

def train_poisoned_rf(X,y,p):
    y_train_poisoned = poison(y, p)
    rf_clf = RandomForestClassifier(n_estimators=25, max_features="sqrt", max_depth=None)
    rf_clf.fit(X_train_transformed, y_train_poisoned)

    return rf_clf

data_train = pd.read_csv("combined_data_train.csv")
data_test = pd.read_csv("combined_data_test.csv")

X_train,y_train = data_train[["h_hitrate", "sentiment", "lifespan"]], data_train["label"]
y_train = y_train.map({"troll": 1, "baseline": 0})

X_test,y_test = data_test[["h_hitrate", "sentiment", "lifespan"]], data_test["label"]
y_test = y_test.map({"troll": 1, "baseline": 0})


y_train_poisoned_25 = poison(y_train, 0.25)
y_train_poisoned_50 = poison(y_train, 0.50)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)

rf_clf = train_poisoned_rf(X_train_transformed, y_train, 0)
rf_y_test_pred = rf_clf.predict(X_test_transformed)
rf_test_report = classification_report(y_test, rf_y_test_pred)
print("RF test report at p=0.00")
print(rf_test_report)
print()

rf_clf = train_poisoned_rf(X_train_transformed, y_train, 0.10)
rf_y_test_pred = rf_clf.predict(X_test_transformed)
rf_test_report = classification_report(y_test, rf_y_test_pred)
print("RF test report at p=0.10")
print(rf_test_report)
print()

rf_clf = train_poisoned_rf(X_train_transformed, y_train, 0.25)
rf_y_test_pred = rf_clf.predict(X_test_transformed)
rf_test_report = classification_report(y_test, rf_y_test_pred)
print("RF test report at p=0.25")
print(rf_test_report)
print()

rf_clf = train_poisoned_rf(X_train_transformed, y_train, 0.50)
rf_y_test_pred = rf_clf.predict(X_test_transformed)
rf_test_report = classification_report(y_test, rf_y_test_pred)
print("RF test report at p=0.50")
print(rf_test_report)
print()    

ps = np.linspace(0,1,50)
accss = []
for _ in range(10):
    accs = []
    for p in ps:
        rf_clf = train_poisoned_rf(X_train_transformed, y_train, p)
        rf_y_test_pred = rf_clf.predict(X_test_transformed)
        acc = accuracy_score(y_test, rf_y_test_pred)
        accs.append(acc)
    accss.append(accs)

means = np.array(accss).mean(axis=0)
stds = np.array(accss).std(axis=0)

plt.plot(ps, means, color='c', label="empirical accuracy development over p (10 run avg.)")
plt.plot(ps, means+stds, color='c', linestyle='dashed', alpha=0.5, label="standard deviation")
plt.plot(ps, means-stds, color='c', linestyle='dashed', alpha=0.5)
plt.xlabel("p (poisoning rate)")
plt.ylabel("model accuracy")
plt.legend()
plt.show()
