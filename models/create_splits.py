import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
data["label"] = "troll"
baseline["label"] = "baseline"

troll_authors = data["author"].unique()
baseline_authors = baseline["author"].unique()

troll_authors_train, troll_authors_test = train_test_split(troll_authors, test_size=0.50, shuffle=True, random_state=12345)
troll_authors_test, troll_authors_val = train_test_split(troll_authors_test, test_size=0.50, shuffle=True, random_state=12345)
baseline_authors_train, baseline_authors_test = train_test_split(baseline_authors, test_size=0.50, shuffle=True, random_state=12345)
baseline_authors_test, baseline_authors_val = train_test_split(baseline_authors_test, test_size=0.50, shuffle=True, random_state=12345)

data_train = data[data["author"].isin(troll_authors_train)]
data_val = data[data["author"].isin(troll_authors_val)]
data_test = data[data["author"].isin(troll_authors_test)]

baseline_train = baseline[baseline["author"].isin(baseline_authors_train)]
baseline_val = baseline[baseline["author"].isin(baseline_authors_val)]
baseline_test = baseline[baseline["author"].isin(baseline_authors_test)]

train_set = pd.concat([data_train, baseline_train], join="inner")
val_set = pd.concat([data_val, baseline_val], join="inner")
test_set = pd.concat([data_test, baseline_test], join="inner")

pd.DataFrame(troll_authors_train).to_csv("troll_authors_train.csv", index=False)
pd.DataFrame(troll_authors_val).to_csv("troll_authors_val.csv", index=False)
pd.DataFrame(troll_authors_test).to_csv("troll_authors_test.csv", index=False)
pd.DataFrame(baseline_authors_train).to_csv("baseline_authors_train.csv", index=False)
pd.DataFrame(baseline_authors_val).to_csv("baseline_authors_val.csv", index=False)
pd.DataFrame(baseline_authors_test).to_csv("baseline_authors_test.csv", index=False)
