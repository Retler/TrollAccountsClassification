"""
This script read the information of random sampled users, their age and followers/following count
This information is aggregated to calculate the average follower/following progression over time
"""
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy

DATESET = "user_data.csv"

data = pd.read_csv(DATESET, header=None, names=["author", "created_at", "followers", "following"])
data.drop_duplicates(subset=['author'], inplace=True)

data["age_weeks"] = data["created_at"].apply(lambda x: int((datetime.today() - datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.000Z")).days / 7) )

ten_year_accounts = data[data["age_weeks"] <= 520]
plt.plot(ten_year_accounts.groupby("age_weeks").count()["author"])
plt.xlabel("Account age in weeks")
plt.ylabel("Number of accounts")
plt.show() # The high number of young accounts can be explained by bots or inauthentic accounts which are closed over time

group = ten_year_accounts.groupby(["age_weeks"])["followers"].mean()
x = group.keys()
y = group.values
model = numpy.poly1d(numpy.polyfit(x, y, 3))
plt.plot(group, label="Follower avg.")
plt.plot(x, model(x), label="regression")
plt.xlabel("Account age in weeks")
plt.ylabel("Average number of followers")
plt.show()
