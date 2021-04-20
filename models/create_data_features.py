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

### Hashtag similarities ###
data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(30)
baseline["hashtags"] = baseline["content"].apply(lambda x: preprocess(str(x), is_hashtag))
baseline_hashtag_vocabulary = baseline["hashtags"].explode().value_counts().sort_values().tail(30)
troll_vs_troll_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_hashtag_hitrate = baseline.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

### Sentiment  ###
troll_sentiment = data.groupby("author")["sentiment"].mean()
baseline_sentiment = baseline.groupby("author")["sentiment"].mean()

### Activity variability ### TODO: FIX ME (look at variability only during active periods)
std_troll_weekly_activity = data.groupby(["week", "author"])["content"].count().groupby("week").median().std()
std_baseline_weekly_activity = baseline.groupby(["week", "author"])["content"].count().groupby("week").median().std()

### Lifespan ###
troll_activity_days = (data.groupby("author")["publish_date"].max() - data.groupby("author")["publish_date"].min()).apply(lambda x: x.days)
baseline_activity_days = (baseline.groupby("author")["publish_date"].max() - baseline.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

