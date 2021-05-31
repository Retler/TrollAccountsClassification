import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import sys
sys.path.append('./data')
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

### Import data ###
data = pd.read_csv("./data/tweets_full.csv", parse_dates=["publish_date"])

data.rename(columns={"publish_date": "datetime"}, inplace=True)
data["date"] = data["datetime"].apply(lambda x: x.date())
data["week"] = data["datetime"].apply(lambda x: x.week)
start_2016 = datetime.strptime("2016-01-01", "%Y-%m-%d").date()
end_2016 = datetime.strptime("2016-12-31", "%Y-%m-%d").date()
end_2016 = datetime.strptime("2016-12-31", "%Y-%m-%d").date()
data = data[data["date"] >= start_2016]
data = data[data["date"] <= end_2016]
data = data[data["account_category"] != "Commercial"]
data = data[data["account_category"] != "NonEnglish"]
data = data[data["account_category"] != "Unknown"]

left_right_filter = np.logical_or(data["account_category"] == "LeftTroll", data["account_category"] == "RightTroll") 
left_right = data[left_right_filter]
other = data[np.logical_not(left_right_filter)]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Week', x=0.5, y=0.02, verticalalignment='bottom', horizontalalignment='center', fontsize=15, fontweight='bold')
ax2.plot(other.groupby("week")["author"].count(), linewidth=1, zorder=2, label="Tweets")
ax2.set_title("Fearmonger, HashtagGamer, NewsFeed", fontsize='15', fontweight='bold')
week_of_election = 45
week_of_dnc_hack_and_pussygate = 40
week_of_hillary_fainting = 36

ax2.axvline(x=week_of_election, color="green", linewidth=2, label="Election", zorder=1, alpha=0.5, linestyle='--')
ax2.axvline(x=week_of_dnc_hack_and_pussygate, color='r', linewidth=2, label="Pussygate", zorder=1, alpha=0.5, linestyle='--')
ax2.axvline(x=week_of_hillary_fainting, color='m', linewidth=2, label="Hillary faint", zorder=1, alpha=0.5, linestyle='--')
ax2.legend(fontsize=13)

ax1.plot(left_right.groupby("week")["author"].count(), linewidth=1, zorder=2, label="Tweets")
ax1.set_title("LeftTroll, RightTroll", fontsize='15', fontweight='bold')

ax1.axvline(x=week_of_election, color="green", linewidth=2, label="Election", zorder=1, alpha=0.5, linestyle='--')
ax1.axvline(x=week_of_dnc_hack_and_pussygate, color='r', linewidth=2, label="Pussygate", zorder=1, alpha=0.5, linestyle='--')
ax1.axvline(x=week_of_hillary_fainting, color='m', linewidth=2, label="Hillary faint", zorder=1, alpha=0.5, linestyle='--')
ax1.legend(fontsize=13)
ylim = ax1.get_ylim()
ax2.set_ylim(ylim)
ax1.set_ylabel("Tweet frequency")

plt.show()
