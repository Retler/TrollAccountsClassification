import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Timezone of troll tweets is in UTC: https://github.com/fivethirtyeight/russian-troll-tweets/issues/9
# The same goes for baseline_dataset (Twitter API returns UTC time per default)

### Import data ###
data = pd.read_csv("tweets_full.csv", parse_dates=["date"])

baseline_data = pd.read_csv("baseline_dataset.csv", parse_dates=["publish_date"])
baseline_data.rename(columns={"publish_date": "datetime"}, inplace=True)
baseline_data["date"] = baseline_data["datetime"].apply(lambda x: x.date())
data.rename(columns={"date": "datetime"}, inplace=True)
data["date"] = data["datetime"].apply(lambda x: x.date())

### Plot account category counts ###
# account_categories_groups = data.groupby("account_category").agg("count").iloc[:,0]
# account_categories = account_categories_groups.keys()
# account_categories_counts = account_categories_groups.values

# plt.bar(np.arange(len(account_categories)), account_categories_counts)
# plt.xticks(np.arange(len(account_categories)), account_categories)

# plt.show()

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

data["hour"] = data["datetime"].apply(lambda x: x.hour)
hour_bins = data.groupby("hour")["hour"].count()
plt.bar(np.arange(len(hour_bins)), hour_bins)
plt.xticks(np.arange(len(hour_bins)), np.arange(len(hour_bins)))
plt.xlabel("hour of day")
plt.ylabel("tweets")

plt.show()

### Plot 'date' hour distribution for baseline ###

baseline_data["hour"] = baseline_data["datetime"].apply(lambda x: x.hour)
hour_bins = baseline_data.groupby("hour")["hour"].count()
plt.bar(np.arange(len(hour_bins)), hour_bins)
plt.xticks(np.arange(len(hour_bins)), np.arange(len(hour_bins)))
plt.xlabel("hour of day")
plt.ylabel("tweets")

plt.show()

### Plot average tweets per day per category ###


#active_days = data.groupby(["author", "account_category"])["date"].max() -  data.groupby(["author", "account_category"])["date"].min()
#active_days = active_days.apply(lambda x: x.days) + 1 # Add one to remove division by 0 errors
#tweet_counts_authors = data.groupby(["author", "account_category"])["author"].count()
#avg_tweets_per_day = (tweet_counts_authors / active_days).sum() / len(tweet_counts_authors)

#tweets_author_day = tweet_counts_authors / active_days
#tweets_day_category = tweets_author_day.groupby("account_category").sum() / tweets_author_day.groupby("account_category").count()


### Plot development of tweets over time (avg. per day), this can be an indicator of collaboration ###
