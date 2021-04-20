import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

troll_activity_days = (data.groupby("author")["publish_date"].max() - data.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
baseline_activity_days = (baseline.groupby("author")["publish_date"].max() - baseline.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

plt.subplot(1, 2, 1)
troll_activity_cdf = troll_activity_days.value_counts().sort_index().cumsum()
(troll_activity_cdf / troll_activity_days.size).plot(label="Troll")
baseline_activity_cdf = baseline_activity_days.value_counts().sort_index().cumsum()
(baseline_activity_cdf / baseline_activity_days.size).plot(label="Baseline")
plt.xlabel("Lifespan in days")
plt.ylabel("Account share")
plt.title("CDF of the account lifespan in 2016")
plt.legend()

plt.subplot(1, 2, 2)
data["month"] = data["publish_date"].apply(lambda x: x.month)
baseline["month"] = baseline["publish_date"].apply(lambda x: x.month)
(data.groupby("month")["author"].unique().apply(len) / data["author"].unique().size).plot(label="Troll")
(baseline.groupby("month")["author"].unique().apply(len) / baseline["author"].unique().size).plot(label="Baseline")
plt.xticks(np.arange(12) + 1, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"], rotation=0.35)
plt.xlabel("Month")
plt.ylabel("Active accounts share")
plt.title("Active accounts per month in 2016")
plt.legend()

plt.show()
