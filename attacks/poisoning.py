from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

def poison(y, p):
    return [(1-s) if np.random.rand() < p else s for s in y]

def train_poisoned_clf(X,y,p,clf):
    y_train_poisoned = poison(y, p)
    clf.fit(X, y_train_poisoned)

    return clf

data_train = pd.read_csv("combined_data_train.csv")
data_test = pd.read_csv("combined_data_test.csv")
data_test = pd.read_csv("combined_data_val.csv")

X_train,y_train = data_train[["h_hitrate", "sentiment", "lifespan", "subjectivity"]], data_train["label"]
y_train = y_train.map({"troll": 1, "baseline": 0})

X_val,y_val = data_test[["h_hitrate", "sentiment", "lifespan", "subjectivity"]], data_test["label"]
y_val = y_val.map({"troll": 1, "baseline": 0})

X_test,y_test = data_test[["h_hitrate", "sentiment", "lifespan", "subjectivity"]], data_test["label"]
y_test = y_test.map({"troll": 1, "baseline": 0})

scaler = StandardScaler()
scaler.fit(X_train)
X_train_transformed = scaler.transform(X_train)
X_test_transformed = scaler.transform(X_test)
X_val_transformed = scaler.transform(X_val)

clf = train_poisoned_clf(X_train_transformed, y_train, 0, RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=80))
y_test_pred = clf.predict(X_test_transformed)
rf_report = classification_report(y_test, y_test_pred)
print("RF report for p=0")
print(rf_report)
print()

clf = train_poisoned_clf(X_train_transformed, y_train, 0.1, RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=80))
y_test_pred = clf.predict(X_test_transformed)
rf_report = classification_report(y_test, y_test_pred)
print("RF report for p=0.1")
print(rf_report)
print()

clf = train_poisoned_clf(X_train_transformed, y_train, 0.25, RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=80))
y_test_pred = clf.predict(X_test_transformed)
rf_report = classification_report(y_test, y_test_pred)
print("RF report for p=0.25")
print(rf_report)
print()

clf = train_poisoned_clf(X_train_transformed, y_train, 0.5, RandomForestClassifier(n_estimators=50, max_features="sqrt", max_depth=80))
y_test_pred = clf.predict(X_test_transformed)
rf_report = classification_report(y_test, y_test_pred)
print("RF report for p=0.5")
print(rf_report)
print()

models = [RandomForestClassifier(n_estimators=50, max_depth=80, max_features='sqrt'), KNeighborsClassifier(n_neighbors=200, weights='uniform'), svm.SVC(C=0.1, gamma=0.01, kernel='sigmoid')]
ps = np.linspace(0,1,50)
meanss = []
stdss = []

for m in models:
    accss = []
    for _ in range(10):
        accs = []
        for p in ps:
            clf = train_poisoned_clf(X_train_transformed, y_train, p, m)
            val_pred = clf.predict(X_val_transformed)
            acc = accuracy_score(y_val, val_pred)
            accs.append(acc)
        accss.append(accs)

    meanss.append(np.array(accss).mean(axis=0))
    stdss.append(np.array(accss).std(axis=0))

colors = ['c', 'm', 'y']
clf_names = ["Random Forest", "KNN", "SVM"]
for means,stds,m,c,n in zip(meanss, stdss, models, colors, clf_names):
    plt.plot(ps, means, color=c, label=n)

plt.xlabel("p (poisoning rate)")
plt.ylabel("model accuracy")
plt.legend()
plt.show()
