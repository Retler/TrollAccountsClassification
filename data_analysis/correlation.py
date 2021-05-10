import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from scipy.stats import pointbiserialr
plt.style.use('seaborn')
plt.rcParams['font.size'] = '20'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

combined_data = glob.glob("./combined_data_*.csv")
data = pd.concat((pd.read_csv(f) for f in combined_data))
data_full = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline_full = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
data["label"] = data["label"].map({"troll": 1, "baseline": 0})

### Sentiment ###
cor_sentiment = pointbiserialr(data["label"], data["sentiment"])
print("Sentiment vs Account type correlation:")
print(cor_sentiment)

### Lifespan ###
cor_lifespan = pointbiserialr(data["label"], data["lifespan"])
print("Account lifespan vs Account type correlation:")
print(cor_lifespan)

### follower/following ratio ###
cor_f_ratio = pointbiserialr(data["label"], data["f_ratio"])
print("Follower/Following ratio vs Account type correlation:")
print(cor_f_ratio)

### Hashtag hitrate ###
cor_h_hitrate = pointbiserialr(data["label"], data["h_hitrate"])
print("Hashtag hitrate vs Account type correlation:")
print(cor_h_hitrate)

### Retweet rate ###
baseline_retweet_mean = baseline_full.groupby("author")["retweet"].mean().to_frame()
troll_retweet_mean = data_full.groupby("author")["retweet"].mean().to_frame()
baseline_retweet_mean["label"] = 0
troll_retweet_mean["label"] = 1
retweets = pd.concat([baseline_retweet_mean, troll_retweet_mean])
cor_retweet = pointbiserialr(retweets["label"], retweets["retweet"])

### Subjectivity score ###
subjectivities_baseline = baseline_full.groupby("author")["subjectivity"].mean().to_frame()
subjectivities_troll = data_full.groupby("author")["subjectivity"].mean().to_frame()
subjectivities_baseline["label"] = 0
subjectivities_troll["label"] = 1
subjectivities = pd.concat([subjectivities_baseline, subjectivities_troll])
cor_subjectivity = pointbiserialr(subjectivities["label"], subjectivities["subjectivity"])
