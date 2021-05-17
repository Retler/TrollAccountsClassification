import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

troll_follower_following = data.groupby("author")["followers"].mean() / (data.groupby("author")["following"].mean() + 1)
troll_follower_following = troll_follower_following.sort_values()[:-25]
baseline_follower_following = baseline.groupby("author")["followers"].mean() / (baseline.groupby("author")["following"].mean() + 1)

weights_baseline = np.ones(len(baseline_follower_following)) / len(baseline_follower_following)
weights_troll = np.ones(len(troll_follower_following)) / len(troll_follower_following)

plt.subplot(1,2,1)
plt.hist([baseline_follower_following, troll_follower_following], weights=[weights_baseline, weights_troll], bins=35, label=["baseline", "troll"])
plt.legend()
plt.xlabel("follower/following rate")
plt.ylabel("density")
plt.title("Distributions of the follower/following rates", fontsize=15, fontweight='bold')

plt.subplot(1,2,2)
labels = ["followers", "following"]
troll_bars = [data.groupby("author")["followers"].mean().mean(), data.groupby("author")["following"].mean().mean()]
baseline_bars = [baseline.groupby("author")["followers"].mean().mean(), baseline.groupby("author")["following"].mean().mean()]
x = np.arange(len(labels))
width = 0.20
plt.bar(x - width/2, baseline_bars, width, label='baseline')
plt.bar(x + width/2, troll_bars, width, label='troll')
plt.ylabel("Count")
plt.xticks(x, labels=labels)
plt.legend()
plt.title("Average follower and following count per account", fontsize=15, fontweight='bold')

plt.show()
