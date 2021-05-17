import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('./data')
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

# Timezone of troll tweets is in UTC: https://github.com/fivethirtyeight/russian-troll-tweets/issues/9
# The same goes for baseline_dataset (Twitter API returns UTC time per default)

### Import data ###
data = pd.read_csv("./data/tweets_full.csv", parse_dates=["publish_date"], nrows=100000)

#baseline_data = pd.read_csv("baseline_dataset_english.csv", parse_dates=["publish_date"])
#baseline_data.rename(columns={"publish_date": "datetime"}, inplace=True)
#baseline_data["date"] = baseline_data["datetime"].apply(lambda x: x.date())
data["week"] = data["publish_date"].apply(lambda x: x.week)
data["date"] = data["publish_date"].apply(lambda x: x.date())
start_2016 = datetime.strptime("2016-01-01", "%Y-%m-%d").date()
end_2016 = datetime.strptime("2016-12-31", "%Y-%m-%d").date()
data = data[data["date"] >= start_2016]
data = data[data["date"] <= end_2016]
data = data[data["account_category"] != "Commercial"]
data = data[data["account_category"] != "NonEnglish"]
data = data[data["account_category"] != "Unknown"]

### Plot account category counts ###
account_categories_groups = data.groupby("account_category")["author"].count()
account_categories = account_categories_groups.keys()
account_categories_counts = account_categories_groups.values

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Account category', x=0.5, y=0.02, verticalalignment='bottom', horizontalalignment='center', fontsize=15, fontweight='bold')
ax1.bar(np.arange(len(account_categories)), account_categories_counts)
ax1.set_xticks(np.arange(len(account_categories)))
ax1.set_xticklabels(account_categories, rotation=10)
ax1.set_ylabel("Number of tweets")

account_categories_groups = data.groupby("account_category")["author"].unique().apply(len)
account_categories = account_categories_groups.keys()
account_categories_counts = account_categories_groups.values

ax2.bar(np.arange(len(account_categories)), account_categories_counts)
ax2.set_xticks(np.arange(len(account_categories)))
ax2.set_xticklabels(account_categories, rotation=10)
plt.ylabel("Number of accounts")

plt.show()

data = data[data["account_category"] != "Fearmonger"]
data = data[data["account_category"] != "HashtagGamer"]
data = data[data["account_category"] != "NewsFeed"]

plt.plot(data.groupby("week")["author"].count(), linewidth=1, zorder=2)
plt.xlabel("week")
plt.ylabel("tweet frequency")
week_of_election = 45
week_of_dnc_hack_and_pussygate = 40
week_of_hillary_fainting = 36

plt.axvline(x=week_of_election, color="green", linewidth=1, label="Election", zorder=1, alpha=0.5, linestyle='--')
plt.axvline(x=week_of_dnc_hack_and_pussygate, color='r', linewidth=1, label="Pussygate", zorder=1, alpha=0.5, linestyle='--')
plt.axvline(x=week_of_hillary_fainting, color='m', linewidth=1, label="Hillary faint", zorder=1, alpha=0.5, linestyle='--')
plt.legend()

plt.show()

# ### Plot 'followers'/'following' ratio ###
# following = data.groupby("account_category")["following"].mean()
# followers = data.groupby("account_category")["followers"].mean()

# ratio = followers / following
# plt.bar(np.arange(len(ratio)), ratio,  label="followers/following ratio")
# plt.xticks(np.arange(len(ratio)), followers.keys())
# plt.hlines(1, xmin=0, xmax=len(ratio), linestyles='dashed', colors='k', linewidths=1, label="1.0")
# plt.legend()
# plt.show()

# ### Plot means and standard deviation of 'content' length ### 
# categories = data["account_category"].unique()

# means = []
# stds = []
# x_pos = np.arange(len(categories))
# for c in categories:
#     means.append(data[data["account_category"] == c]["content"].apply(len).mean())
#     stds.append(data[data["account_category"] == c]["content"].apply(len).std())

# plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10)
# plt.xticks(x_pos, categories)
# plt.xlabel("Account category")
# plt.ylabel("Tweet length")
# plt.show()

# ### Plot retweet share ### 
# categories = data["account_category"].unique()

# retweet_shares = []
# x_pos = np.arange(len(categories))
# for c in categories:
#     retweet_shares.append(data["retweet"].sum() / data["retweet"].count())
    
# plt.bar(x_pos, retweet_shares, alpha=0.5, ecolor='black', capsize=10)
# plt.xticks(x_pos, categories)
# plt.xlabel("Account category")
# plt.ylabel("Retweet percentage")
# print(retweet_shares)
# plt.show()

### Plot 'date' hour distribution ###

# data["hour"] = data["datetime"].apply(lambda x: x.hour)
# hour_bins = data.groupby("hour")["hour"].count()
# plt.bar(np.arange(len(hour_bins)), hour_bins)
# plt.xticks(np.arange(len(hour_bins)), np.arange(len(hour_bins)))
# plt.xlabel("hour of day")
# plt.ylabel("tweets")

# plt.show()

### Plot 'date' hour distribution for baseline ###

# baseline_data["hour"] = baseline_data["datetime"].apply(lambda x: x.hour)
# hour_bins = baseline_data.groupby("hour")["hour"].count()
# plt.bar(np.arange(len(hour_bins)), hour_bins)
# plt.xticks(np.arange(len(hour_bins)), np.arange(len(hour_bins)))
# plt.xlabel("hour of day")
# plt.ylabel("tweets")

# plt.show()

### Mean and std of average tweets per day ()

### Plot average tweets per day per category ###


#active_days = data.groupby(["author", "account_category"])["date"].max() -  data.groupby(["author", "account_category"])["date"].min()
#active_days = active_days.apply(lambda x: x.days) + 1 # Add one to remove division by 0 errors
#tweet_counts_authors = data.groupby(["author", "account_category"])["author"].count()
#avg_tweets_per_day = (tweet_counts_authors / active_days).sum() / len(tweet_counts_authors)

#tweets_author_day = tweet_counts_authors / active_days
#tweets_day_category = tweets_author_day.groupby("account_category").sum() / tweets_author_day.groupby("account_category").count()


### Plot development of tweets over time (avg. per day), this can be an indicator of collaboration ###
