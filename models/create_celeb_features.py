import pandas as pd
import numpy as np
import sys
sys.path.append('./data_analysis')
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate

# 1. Hashtag similarities
# 2. Sentiment
# 3. subjectivity
# 4. Lifespan

trolls = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
data = pd.read_csv("celeb_tweets.csv", parse_dates=["publish_date"])

### Hashtag similarities ###

data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
trolls["hashtags"] = trolls["content"].apply(lambda x: preprocess(str(x), is_hashtag))
troll_hashtag_vocabulary = trolls["hashtags"].explode().value_counts().sort_values().tail(30)

celeb_vs_troll_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))

### Sentiment  ###
celeb_sentiment = data.groupby("author")["sentiment"].mean()

### Subjectivity ###
celeb_subjectivity = data.groupby("author")["subjectivity"].mean()

### Lifespan ###
celeb_activity_days = (data.groupby("author")["publish_date"].max() - data.groupby("author")["publish_date"].min()).apply(lambda x: x.days)

### Combine data ###
celeb_features = pd.concat([celeb_vs_troll_hashtag_hitrate.rename("h_hitrate"), celeb_sentiment.rename("sentiment"), celeb_subjectivity.rename("subjectivity"),  celeb_activity_days.rename("lifespan")], axis=1)
