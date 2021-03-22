import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from gensim.parsing.preprocessing import remove_stopwords
plt.style.use('seaborn')

# Timezone of troll tweets is in UTC: https://github.com/fivethirtyeight/russian-troll-tweets/issues/9
# The same goes for baseline_dataset (Twitter API returns UTC time per default)

### Import data ###
data = pd.read_csv("tweets_full.csv", parse_dates=["date"], nrows=10000)
baseline_data = pd.read_csv("baseline_dataset_english.csv", parse_dates=["date"], nrows=10000)

troll_vocabulary = [x for tweet in data["content"] for x in remove_stopwords(str(tweet).lower()).split()]