import pandas as pd
import numpy as np
import sys
sys.path.append('./data_analysis')
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

# 1. Hashtag similarities
# 2. Sentiment
# 3. Activity variability
# 4. Lifespan

# ### Hashtag similarities ###
# data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
# troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(30)
# baseline["hashtags"] = baseline["content"].apply(lambda x: preprocess(str(x), is_hashtag))
# baseline_hashtag_vocabulary = baseline["hashtags"].explode().value_counts().sort_values().tail(30)
# troll_vs_troll_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
# baseline_vs_troll_hashtag_hitrate = baseline.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

# ### Sentiment  ###
# troll_sentiment = data.groupby("author")["sentiment"].mean()
# baseline_sentiment = baseline.groupby("author")["sentiment"].mean()

### Activity variability ### Reject??
#std_troll_activity = data.groupby(["author", "publish_date"])["content"].count().groupby("author").std()
#std_baseline_activity = baseline.groupby(["author", "publish_date"])["content"].count().groupby("author").std()

### TODO: Follower/Following ###

# ### Lifespan ###
# troll_activity_days = (data.groupby("author")["publish_date"].max() - data.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
# baseline_activity_days = (baseline.groupby("author")["publish_date"].max() - baseline.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

### Follower/Following rate ###

# troll_follower_following = data.groupby("author")["followers"].mean() / (data.groupby("author")["following"].mean() + 1)
# baseline_follower_following = baseline.groupby("author")["followers"].mean() / (baseline.groupby("author")["following"].mean() + 1)

# ### Combine data ###
# troll_features = pd.concat([troll_vs_troll_hashtag_hitrate.rename("h_hitrate"), troll_sentiment.rename("sentiment"), troll_activity_d
# ays.rename("lifespan")], axis=1)
# baseline_features = pd.concat([baseline_vs_troll_hashtag_hitrate.rename("h_hitrate"), baseline_sentiment.rename("sentiment"), baseline_activity_days.rename("lifespan")], axis=1)
# troll_features["label"] = "troll"
# baseline_features["label"] = "troll"
# combined_data = pd.concat([troll_features, baseline_features])
