import pandas as pd
import numpy as np
import sys
sys.path.append('./data_analysis')
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate

# 1. Hashtag similarities
# 2. Sentiment
# 3. subjectivity
# 4. Lifespan

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

troll_authors_train = pd.read_csv("./models/splits/troll_authors_train.csv")
troll_authors_val = pd.read_csv("./models/splits/troll_authors_val.csv")
troll_authors_test = pd.read_csv("./models/splits/troll_authors_test.csv")
baseline_authors_train = pd.read_csv("./models/splits/baseline_authors_train.csv")
baseline_authors_val = pd.read_csv("./models/splits/baseline_authors_val.csv")
baseline_authors_test = pd.read_csv("./models/splits/baseline_authors_test.csv")

data_train = data[data["author"].isin(troll_authors_train.values[:,0])]
data_val = data[data["author"].isin(troll_authors_val.values[:,0])]
data_test = data[data["author"].isin(troll_authors_test.values[:,0])]
baseline_train = baseline[baseline["author"].isin(baseline_authors_train.values[:,0])]
baseline_val = baseline[baseline["author"].isin(baseline_authors_val.values[:,0])]
baseline_test = baseline[baseline["author"].isin(baseline_authors_test.values[:,0])]

### Hashtag similarities ###

data_train["hashtags"] = data_train["content"].apply(lambda x: preprocess(str(x), is_hashtag))
data_val["hashtags"] = data_val["content"].apply(lambda x: preprocess(str(x), is_hashtag))
data_test["hashtags"] = data_test["content"].apply(lambda x: preprocess(str(x), is_hashtag))
data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(30)
baseline_train["hashtags"] = baseline_train["content"].apply(lambda x: preprocess(str(x), is_hashtag))
baseline_val["hashtags"] = baseline_val["content"].apply(lambda x: preprocess(str(x), is_hashtag))
baseline_test["hashtags"] = baseline_test["content"].apply(lambda x: preprocess(str(x), is_hashtag))

troll_vs_troll_hashtag_hitrate_train = data_train.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
troll_vs_troll_hashtag_hitrate_val = data_val.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
troll_vs_troll_hashtag_hitrate_test = data_test.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

baseline_vs_troll_hashtag_hitrate_train = baseline_train.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_hashtag_hitrate_val = baseline_val.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_hashtag_hitrate_test = baseline_test.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

### Sentiment  ###
troll_sentiment_train = data_train.groupby("author")["sentiment"].mean()
troll_sentiment_val = data_val.groupby("author")["sentiment"].mean()
troll_sentiment_test = data_test.groupby("author")["sentiment"].mean()
baseline_sentiment_train = baseline_train.groupby("author")["sentiment"].mean()
baseline_sentiment_val = baseline_val.groupby("author")["sentiment"].mean()
baseline_sentiment_test = baseline_test.groupby("author")["sentiment"].mean()

### Subjectivity ###
troll_subjectivity_train = data_train.groupby("author")["subjectivity"].mean()
troll_subjectivity_val = data_val.groupby("author")["subjectivity"].mean()
troll_subjectivity_test = data_test.groupby("author")["subjectivity"].mean()
baseline_subjectivity_train = baseline_train.groupby("author")["subjectivity"].mean()
baseline_subjectivity_val = baseline_val.groupby("author")["subjectivity"].mean()
baseline_subjectivity_test = baseline_test.groupby("author")["subjectivity"].mean()

### Activity variability ### Reject??
std_troll_activity = data.groupby(["author", "publish_date"])["content"].count().groupby("author").std()
std_baseline_activity = baseline.groupby(["author", "publish_date"])["content"].count().groupby("author").std()

### Lifespan ###
troll_activity_days_train = (data_train.groupby("author")["publish_date"].max() - data_train.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
troll_activity_days_val = (data_val.groupby("author")["publish_date"].max() - data_val.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
troll_activity_days_test = (data_test.groupby("author")["publish_date"].max() - data_test.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

baseline_activity_days_train = (baseline_train.groupby("author")["publish_date"].max() - baseline_train.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
baseline_activity_days_val = (baseline_val.groupby("author")["publish_date"].max() - baseline_val.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
baseline_activity_days_test = (baseline_test.groupby("author")["publish_date"].max() - baseline_test.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

### Follower/Following rate ###
troll_follower_following_train = data_train.groupby("author")["followers"].mean() / (data_train.groupby("author")["following"].mean() + 1)
troll_follower_following_val = data_val.groupby("author")["followers"].mean() / (data_val.groupby("author")["following"].mean() + 1)
troll_follower_following_test = data_test.groupby("author")["followers"].mean() / (data_test.groupby("author")["following"].mean() + 1)
baseline_follower_following_train = baseline_train.groupby("author")["followers"].mean() / (baseline_train.groupby("author")["following"].mean() + 1)
baseline_follower_following_val = baseline_val.groupby("author")["followers"].mean() / (baseline_val.groupby("author")["following"].mean() + 1)
baseline_follower_following_test = baseline_test.groupby("author")["followers"].mean() / (baseline_test.groupby("author")["following"].mean() + 1)

### Combine data ###
troll_features_train = pd.concat([troll_vs_troll_hashtag_hitrate_train.rename("h_hitrate"), troll_sentiment_train.rename("sentiment"), troll_subjectivity_train.rename("subjectivity"),  troll_activity_days_train.rename("lifespan"), troll_follower_following_train.rename("f_ratio")], axis=1)
troll_features_val = pd.concat([troll_vs_troll_hashtag_hitrate_val.rename("h_hitrate"), troll_sentiment_val.rename("sentiment"), troll_subjectivity_val.rename("subjectivity"), troll_activity_days_val.rename("lifespan"), troll_follower_following_val.rename("f_ratio")], axis=1)
troll_features_test = pd.concat([troll_vs_troll_hashtag_hitrate_test.rename("h_hitrate"), troll_sentiment_test.rename("sentiment"), troll_subjectivity_test.rename("subjectivity"), troll_activity_days_test.rename("lifespan"), troll_follower_following_test.rename("f_ratio")], axis=1)

baseline_features_train = pd.concat([baseline_vs_troll_hashtag_hitrate_train.rename("h_hitrate"), baseline_sentiment_train.rename("sentiment"), baseline_subjectivity_train.rename("subjectivity"), baseline_activity_days_train.rename("lifespan"), baseline_follower_following_train.rename("f_ratio")], axis=1)
baseline_features_val = pd.concat([baseline_vs_troll_hashtag_hitrate_val.rename("h_hitrate"), baseline_sentiment_val.rename("sentiment"), baseline_subjectivity_val.rename("subjectivity"), baseline_activity_days_val.rename("lifespan"), baseline_follower_following_val.rename("f_ratio")], axis=1)
baseline_features_test = pd.concat([baseline_vs_troll_hashtag_hitrate_test.rename("h_hitrate"), baseline_sentiment_test.rename("sentiment"), baseline_subjectivity_test.rename("subjectivity"), baseline_activity_days_test.rename("lifespan"), baseline_follower_following_test.rename("f_ratio")], axis=1)

troll_features_train["label"] = "troll"
troll_features_val["label"] = "troll"
troll_features_test["label"] = "troll"

baseline_features_train["label"] = "baseline"
baseline_features_val["label"] = "baseline"
baseline_features_test["label"] = "baseline"

combined_data_train = pd.concat([troll_features_train, baseline_features_train])
combined_data_val = pd.concat([troll_features_val, baseline_features_val])
combined_data_test = pd.concat([troll_features_test, baseline_features_test])

combined_data_train.to_csv("combined_data_train.csv", index=False)
combined_data_val.to_csv("combined_data_val.csv", index=False)
combined_data_test.to_csv("combined_data_test.csv", index=False)
