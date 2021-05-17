import pandas as pd
import numpy as np
import sys
sys.path.append('./data_analysis')
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate
from sklearn.model_selection import train_test_split
print("reading data")


data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
data["publish_date"] = pd.to_datetime(data["publish_date"], utc=True)
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

troll_authors_test = pd.read_csv("./models/splits/troll_authors_test.csv")
baseline_authors_test = pd.read_csv("./models/splits/baseline_authors_test.csv")

data["injected"] = 0
data_test = data[data["author"].isin(troll_authors_test.values[:,0])]
baseline_test = baseline[baseline["author"].isin(baseline_authors_test.values[:,0])]

def enrich_troll_with_baseline_tweets(user, amount):
    global data_test
    global baseline_test
    sampled_rows = baseline_test.sample(amount)
    sampled_rows["author"] = user
    sampled_rows["injected"] = 1
    data_test = pd.concat([data_test, sampled_rows], join="inner")

for author, amount in data_test.groupby("author")["content"].count().iteritems():
    enrich_troll_with_baseline_tweets(author, amount)

print("calculating hashtag similarities")
### Hashtag similarities ###
data_test["hashtags"] = data_test["content"].apply(lambda x: preprocess(str(x), is_hashtag))
data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))

troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(30)
baseline_test["hashtags"] = baseline_test["content"].apply(lambda x: preprocess(str(x), is_hashtag))

troll_vs_troll_hashtag_hitrate_test = data_test.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_hashtag_hitrate_test = baseline_test.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

print("calculating sentiment")
### Sentiment  ###
troll_sentiment_test = data_test.groupby("author")["sentiment"].mean()
baseline_sentiment_test = baseline_test.groupby("author")["sentiment"].mean()

### Subjectivity  ###
troll_subjectivity_test = data_test.groupby("author")["subjectivity"].mean()
baseline_subjectivity_test = baseline_test.groupby("author")["subjectivity"].mean()

print("calculating lifespan")
### Lifespan ###
troll_activity_days_test = (data_test.groupby("author")["publish_date"].max() - data_test.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
baseline_activity_days_test = (baseline_test.groupby("author")["publish_date"].max() - baseline_test.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

print("calculating follower/following rate")
### Follower/Following rate ###
troll_follower_following_test = data_test[data_test["injected"] == 0].groupby("author")["followers"].mean() / (data_test[data_test["injected"] == 0].groupby("author")["following"].mean() + 1)
baseline_follower_following_test = baseline_test.groupby("author")["followers"].mean() / (baseline_test.groupby("author")["following"].mean() + 1)

print("combining and writing data")
### Combine data ###
troll_features_test = pd.concat([troll_vs_troll_hashtag_hitrate_test.rename("h_hitrate"), troll_sentiment_test.rename("sentiment"), troll_subjectivity_test.rename("subjectivity"), troll_activity_days_test.rename("lifespan"), troll_follower_following_test.rename("f_ratio")], axis=1)
baseline_features_test = pd.concat([baseline_vs_troll_hashtag_hitrate_test.rename("h_hitrate"), baseline_sentiment_test.rename("sentiment"), baseline_subjectivity_test.rename("subjectivity"), baseline_activity_days_test.rename("lifespan"), baseline_follower_following_test.rename("f_ratio")], axis=1)
troll_features_test["label"] = "troll"
baseline_features_test["label"] = "baseline"
combined_data_test = pd.concat([troll_features_test, baseline_features_test])
combined_data_test.to_csv("combined_data_evasion_test.csv", index=False)
