import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn')
plt.rcParams['font.size'] = '20'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n')
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n')

# data["subjectivity"] = data["content"].apply(lambda x: TextBlob(str(x)).subjectivity)
# baseline["subjectivity"] = baseline["content"].apply(lambda x: TextBlob(str(x)).subjectivity)

subjectivities_baseline = baseline.groupby("author")["subjectivity"].mean().sort_values()
y_baseline = subjectivities_baseline.values
subjectivities_troll = data.groupby("author")["subjectivity"].mean().sort_values()
y_troll = subjectivities_troll.values

plt.hist([y_troll, y_baseline], bins=50, label=["Troll", "Baseline"])
plt.xlabel("Subjectivity score")
plt.ylabel("Density")
plt.legend()
plt.show()
