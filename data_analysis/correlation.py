import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from scipy.stats import pointbiserialr
plt.style.use('seaborn')
plt.rcParams['font.size'] = '20'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
all_files = glob.glob("./combined_data_*.csv")

data = pd.concat((pd.read_csv(f) for f in all_files))

data["label"] = data["label"].map({"troll": 1, "baseline": 0})
cor_sentiment = pointbiserialr(data["label"], data["sentiment"])
cor_lifespan = pointbiserialr(data["label"], data["lifespan"])
cor_f_ratio = pointbiserialr(data["label"], data["f_ratio"])
cor_h_hitrate = pointbiserialr(data["label"], data["h_hitrate"])

print("Sentiment vs Account type correlation:")
print(cor_sentiment)

print("Account lifespan vs Account type correlation:")
print(cor_lifespan)

print("Follower/Following ratio vs Account type correlation:")
print(cor_f_ratio)

print("Hashtag hitrate vs Account type correlation:")
print(cor_h_hitrate)
