import sys
sys.path.append('./data_analysis')
import pandas as pd
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate

trolls = pd.read_csv("troll_data_2016_english.csv", usecols=["content", "author"], lineterminator='\n')
data = pd.read_csv("./data/trump_tweets.csv", parse_dates=["date"])
data["date2"] = data["date"].apply(lambda x: x.date())
start_2016 = datetime.strptime("2016-01-01", "%Y-%m-%d").date()
end_2016 = datetime.strptime("2016-12-31", "%Y-%m-%d").date()
data = data[data["date2"] >= start_2016]
data = data[data["date2"] <= end_2016]

sa = SentimentIntensityAnalyzer()
sentiment = data["text"].apply(lambda x: sa.polarity_scores(str(x))["compound"]).mean()
subjectivity = data["text"].apply(lambda x: TextBlob(str(x)).subjectivity).mean()
trump_activity_days = (data["date"].max() - data["date"].min()).days

trolls["hashtags"] = trolls["content"].apply(lambda x: preprocess(str(x), is_hashtag))
troll_hashtag_vocabulary = trolls["hashtags"].explode().value_counts().sort_values().tail(30)
hashtags = data["text"].apply(lambda x: preprocess(str(x), is_hashtag))
trump_vs_troll_hashtag_hitrate = hit_rate(troll_hashtag_vocabulary, hashtags.sum())

df = pd.DataFrame([[trump_vs_troll_hashtag_hitrate, sentiment, subjectivity, trump_activity_days, "baseline", "realDonaldTrump"]], columns=["h_hitrate", "sentiment", "subjectivity", "lifespan", "label", "account"])
