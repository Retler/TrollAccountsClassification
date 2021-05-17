import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

data = pd.read_csv("troll_data_2016_english.csv", lineterminator='\n')
baseline = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n')

### Save sentiment scores
#sa = SentimentIntensityAnalyzer()
#data["sentiment"] = data["content"].apply(lambda x: sa.polarity_scores(str(x))["compound"])
#data.to_csv("troll_data_2016_english.csv", index=False)

data.groupby("week")["sentiment"].mean().plot(label="troll")
baseline.groupby("week")["sentiment"].mean().plot(label="baseline")
plt.xlabel("Week")
plt.ylabel("Average sentiment")
plt.legend()
plt.show()
