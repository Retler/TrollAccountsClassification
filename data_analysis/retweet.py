import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')
plt.rcParams['font.size'] = '20'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', parse_dates=["publish_date"])

baseline_retweet_mean = baseline.groupby("author")["retweet"].mean().mean()
troll_retweet_mean = data.groupby("author")["retweet"].mean().mean()


