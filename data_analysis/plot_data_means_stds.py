import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

### Import data ###
data = pd.read_csv("tweets_full.csv", parse_dates=["date"])

data.rename(columns={"date": "datetime"}, inplace=True)
data["date"] = data["datetime"].apply(lambda x: x.date())
data["week"] = data["datetime"].apply(lambda x: x.week)
start_2016 = datetime.strptime("2016-01-01", "%Y-%m-%d").date()
end_2016 = datetime.strptime("2016-12-31", "%Y-%m-%d").date()
data = data[data["date"] >= start_2016]
data = data[data["date"] <= end_2016]
data = data[data["account_category"] != "Commercial"]
data = data[data["account_category"] != "NonEnglish"]
data = data[data["account_category"] != "Unknown"]
data = data[data["account_category"] != "Fearmonger"]
data = data[data["account_category"] != "HashtagGamer"]
data = data[data["account_category"] != "NewsFeed"]

# Tweets per day (measuring only active periods to account for abandoned accounts)
active_days = data.groupby("author")["date"].max() -  data.groupby("author")["date"].min()
active_days = active_days.apply(lambda x: x.days) + 1 # Add one to remove division by 0 errors
tweet_counts_day = data.groupby("author")["author"].count() / active_days

avg_tweets_per_day = tweet_counts_day.sum() / len(tweet_counts_day)
std_tweets_per_day = np.sqrt((np.power(tweet_counts_day - avg_tweets_per_day, 2)).sum() / len(tweet_counts_day))

# Followers 
avg_followers = data.groupby("author")["followers"].max().mean()
std_followers = data.groupby("author")["followers"].max().std()

# Following
avg_following = data.groupby("author")["following"].max().mean()
std_following = data.groupby("author")["following"].max().std()

# Content length
avg_tweet_length = data["content"].apply(len).mean()
std_tweet_length = data["content"].apply(len).std()

# Retweet percentage
retweet_percentage = data.groupby("author")["retweet"].sum() / data.groupby("author")["author"].count()
avg_retweet_percentage = retweet_percentage.mean()
std_retweet_percentage = retweet_percentage.std()

# Plot
row_headers = ["Tweets/day", "Followers", "Following", "Content length", "Retweet percentage"]
column_headers = ["avg.", "std."]
rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))

means = [avg_tweets_per_day, avg_followers, avg_following, avg_tweet_length, avg_retweet_percentage]
stds = [std_tweets_per_day, std_followers, std_following, std_tweet_length, std_retweet_percentage]
cell_text = np.column_stack((means, stds))

plt.table(cellText=cell_text.round(decimals=2), rowLabels=row_headers, rowColours=rcolors, rowLoc='right', colColours=ccolors, colLabels=column_headers, loc='center', colWidths=[0.3, 0.3])
plt.axis('off')
plt.show()