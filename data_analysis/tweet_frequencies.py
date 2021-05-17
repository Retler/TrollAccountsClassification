import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

data["month"] = data["publish_date"].apply(lambda x: x.month_name())
baseline["month"] = baseline["publish_date"].apply(lambda x: x.month_name())

std_troll_weekly_activity = data.groupby(["week", "author"])["content"].count().groupby("week").median().std()
std_baseline_weekly_activity = baseline.groupby(["week", "author"])["content"].count().groupby("week").median().std()

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

data.groupby(["month", "author"]).count()["content"].groupby("month").median().reindex(month_order).plot(label="troll")
baseline.groupby(["month", "author"]).count()["content"].groupby("month").median().reindex(month_order).plot(label="baseline")

month_of_election = list.index(month_order, "November")

plt.axvline(x=month_of_election, color="m", linewidth=2, label="Election", zorder=1, alpha=0.5, linestyle='--')
plt.ylabel("Median Tweets/User")
plt.legend()

plt.show()

### Plot tweet by user CDF
tweet_cdf = baseline.groupby("author")["content"].count().sort_values(ascending=False).cumsum().values
tweet_cdf = tweet_cdf / max(tweet_cdf)
n_users = baseline["author"].unique().size
x = (np.arange(n_users) + 1) / n_users

plt.subplot(1,2,1)
plt.xlabel("User share")
plt.ylabel("Tweet share")
plt.axvline(x=0.1, color="green", linewidth=1, zorder=1, alpha=0.5, linestyle='--')
plt.axhline(y=0.69, color="green", linewidth=1, zorder=1, alpha=0.5, linestyle='--')
plt.plot(x, tweet_cdf)

plt.subplot(1,2,2)
baseline.groupby(["week", "author"]).count()["content"].groupby("week").mean()[:-1].plot(label="mean")
baseline.groupby(["week", "author"]).count()["content"].groupby("week").median()[:-1].plot(label="median")
plt.xlabel("Week")
plt.ylabel("Tweets/User")
plt.legend()

plt.show()
