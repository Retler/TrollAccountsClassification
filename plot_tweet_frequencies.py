import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
plt.style.use('seaborn')

### Import data ###
data = pd.read_csv("tweets_full.csv", parse_dates=["date"])

data.rename(columns={"date": "datetime"}, inplace=True)
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

plt.subplot(1, 2, 1)
plt.plot(other.groupby("week")["author"].count(), linewidth=1, zorder=2)
plt.title("Fearmonger, HashtagGamer, NewsFeed")
plt.xlabel("week")
plt.ylabel("tweet frequency")
week_of_election = 45
week_of_dnc_hack_and_pussygate = 40
week_of_hillary_fainting = 36

plt.axvline(x=week_of_election, color="green", linewidth=1, label="Election", zorder=1, alpha=0.5, linestyle='--')
plt.axvline(x=week_of_dnc_hack_and_pussygate, color='r', linewidth=1, label="Pussygate", zorder=1, alpha=0.5, linestyle='--')
plt.axvline(x=week_of_hillary_fainting, color='m', linewidth=1, label="Hillary faint", zorder=1, alpha=0.5, linestyle='--')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(left_right.groupby("week")["author"].count(), linewidth=1, zorder=2)
plt.title("LeftTroll, RightTroll")
plt.xlabel("week")
plt.ylabel("tweet frequency")

plt.axvline(x=week_of_election, color="green", linewidth=1, label="Election", zorder=1, alpha=0.5, linestyle='--')
plt.axvline(x=week_of_dnc_hack_and_pussygate, color='r', linewidth=1, label="Pussygate", zorder=1, alpha=0.5, linestyle='--')
plt.axvline(x=week_of_hillary_fainting, color='m', linewidth=1, label="Hillary faint", zorder=1, alpha=0.5, linestyle='--')
plt.legend()

plt.show()