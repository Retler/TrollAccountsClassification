import pandas as pd
import numpy as np
import sys
sys.path.append('./data_analysis')
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate
from sklearn.model_selection import train_test_split
print("reading data")
data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
data["injected"] = 0
baseline["injected"] = 0

# 1. Hashtag similarities
# 2. Sentiment
# 3. Lifespan
# 4. Follower/Following ratio

# TODO: make splits based on users
print("Creating test train splits and injecting tweets")
### First, create fake inputs for trolls to mimic evasion ###
# We make the splits on non-aggregated data in order to inject benign tweets into the test set
X_troll_train, X_troll_test = train_test_split(data, test_size=0.25, shuffle=True, random_state=12345)
print(f"X_troll shape before enriching: {X_troll_test.shape}")
X_baseline_train, X_baseline_test = train_test_split(baseline, test_size=0.25, shuffle=True, random_state=12345)

def enrich_troll_with_baseline_tweets(user, amount):
    global X_troll_test
    sampled_rows = X_baseline_test.sample(amount)
    sampled_rows["author"] = user
    sampled_rows["injected"] = 1
    X_troll_test = pd.concat([X_troll_test, sampled_rows], join="inner")

for author, amount in X_troll_test.groupby("author")["content"].count().iteritems():
    enrich_troll_with_baseline_tweets(author, amount)

print("calculating hashtag similarities")
### Hashtag similarities ###
X_troll_train["hashtags"] = X_troll_train["content"].apply(lambda x: preprocess(str(x), is_hashtag))
X_troll_test["hashtags"] = X_troll_test["content"].apply(lambda x: preprocess(str(x), is_hashtag))
data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))

troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(30)
X_baseline_test["hashtags"] = X_baseline_test["content"].apply(lambda x: preprocess(str(x), is_hashtag))
X_baseline_train["hashtags"] = X_baseline_train["content"].apply(lambda x: preprocess(str(x), is_hashtag))
troll_vs_troll_hashtag_hitrate_train = X_troll_train.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
troll_vs_troll_hashtag_hitrate_test = X_troll_test.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_hashtag_hitrate_train = X_baseline_train.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_hashtag_hitrate_test = X_baseline_test.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

print("calculating sentiment")
### Sentiment  ###
troll_sentiment_train = X_troll_train.groupby("author")["sentiment"].mean()
troll_sentiment_test = X_troll_test.groupby("author")["sentiment"].mean()
baseline_sentiment_train = X_baseline_train.groupby("author")["sentiment"].mean()
baseline_sentiment_test = X_baseline_test.groupby("author")["sentiment"].mean()

### Activity variability ### Reject??
std_troll_activity = data.groupby(["author", "publish_date"])["content"].count().groupby("author").std()
std_baseline_activity = baseline.groupby(["author", "publish_date"])["content"].count().groupby("author").std()

print("calculating lifespan")
### Lifespan ###
troll_activity_days_train = (X_troll_train.groupby("author")["publish_date"].max() - X_troll_train.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
# Only compute lifetime on tweets which are not injected
troll_activity_days_test = (X_troll_test[X_troll_test["injected"] == 0].groupby("author")["publish_date"].max() - X_troll_test[X_troll_test["injected"] == 0].groupby("author")["publish_date"].min()).apply(lambda x: x.days)

baseline_activity_days_train = (X_baseline_train.groupby("author")["publish_date"].max() - X_baseline_train.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
baseline_activity_days_test = (X_baseline_test.groupby("author")["publish_date"].max() - X_baseline_test.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

print("calculating follower/following rate")
### Follower/Following rate ###
troll_follower_following_train = X_troll_train.groupby("author")["followers"].mean() / (X_troll_train.groupby("author")["following"].mean() + 1)
troll_follower_following_test = X_troll_test[X_troll_test["injected"] == 0].groupby("author")["followers"].mean() / (X_troll_test[X_troll_test["injected"] == 0].groupby("author")["following"].mean() + 1)
baseline_follower_following_train = X_baseline_train.groupby("author")["followers"].mean() / (X_baseline_train.groupby("author")["following"].mean() + 1)
baseline_follower_following_test = X_baseline_test.groupby("author")["followers"].mean() / (X_baseline_test.groupby("author")["following"].mean() + 1)

print("combining and writing data")
### Combine data ###
troll_features_train = pd.concat([troll_vs_troll_hashtag_hitrate_train.rename("h_hitrate"), troll_sentiment_train.rename("sentiment"), troll_activity_days_train.rename("lifespan")], axis=1)
baseline_features_train = pd.concat([baseline_vs_troll_hashtag_hitrate_train.rename("h_hitrate"), baseline_sentiment_train.rename("sentiment"), baseline_activity_days_train.rename("lifespan")], axis=1)
troll_features_train["label"] = "troll"
baseline_features_train["label"] = "baseline"
combined_data_train = pd.concat([troll_features_train, baseline_features_train])
combined_data_train.to_csv("combined_data_evasion_train.csv", index=False)

troll_features_test = pd.concat([troll_vs_troll_hashtag_hitrate_test.rename("h_hitrate"), troll_sentiment_test.rename("sentiment"), troll_activity_days_test.rename("lifespan")], axis=1)
baseline_features_test = pd.concat([baseline_vs_troll_hashtag_hitrate_test.rename("h_hitrate"), baseline_sentiment_test.rename("sentiment"), baseline_activity_days_test.rename("lifespan")], axis=1)
troll_features_test["label"] = "troll"
baseline_features_test["label"] = "baseline"
combined_data_test = pd.concat([troll_features_train, baseline_features_train])
combined_data_test.to_csv("combined_data_evasion_test.csv", index=False)
