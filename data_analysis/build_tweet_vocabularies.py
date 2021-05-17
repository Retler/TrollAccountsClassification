import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
sys.path.append('./data_analysis')
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from vocabulary_helpers import preprocess, is_hashtag, is_not_hashtag, hit_rate

plt.style.use('seaborn')
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelsize'] = '15'
plt.rcParams['xtick.labelsize'] = '15'
plt.rcParams['ytick.labelsize'] = '15'

### Import data ###
data = pd.read_csv("troll_data_2016_english.csv", usecols=["content", "author"], lineterminator='\n')
data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
#data["words"] = data["content"].apply(lambda x: preprocess(str(x), lambda x: not is_hashtag(x)))
troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(20)
#troll_word_vocabulary = data["words"].explode().value_counts().sort_values().tail(20)

print("1")
baseline_data = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', usecols=["author", "content"])
baseline_data["hashtags"] = baseline_data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
print("2")
#baseline_data["words"] = baseline_data["content"].apply(lambda x: preprocess(str(x), is_not_hashtag))
print("3")
baseline_hashtag_vocabulary = baseline_data["hashtags"].explode().value_counts().sort_values().tail(20)
#baseline_word_vocabulary = baseline_data["words"].explode().value_counts().sort_values().tail(20)
print("4")
#troll_vs_troll_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
#troll_authors_words = data.groupby("author")["words"].apply(lambda x: np.concatenate(x.to_numpy()))
#troll_vs_troll_word_hitrate = troll_authors_words.apply(lambda x: hit_rate(troll_word_vocabulary, x))
print("5")
# troll_vs_baseline_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(baseline_hashtag_vocabulary, x))
#troll_vs_baseline_word_hitrate = troll_authors_words.apply(lambda x: hit_rate(baseline_word_vocabulary, x))

print("6")
# baseline_vs_baseline_hashtag_hitrate = baseline_data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(baseline_hashtag_vocabulary, x))
#baseline_authors_words = baseline_data.groupby("author")["words"].apply(lambda x: np.concatenate(x.to_numpy()))
#baseline_vs_baseline_word_hitrate = baseline_authors_words.apply(lambda x: hit_rate(baseline_word_vocabulary, x))
print("7")
# baseline_vs_troll_hashtag_hitrate = baseline_data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
#baseline_vs_troll_word_hitrate = baseline_authors_words.apply(lambda x: hit_rate(troll_word_vocabulary, x))

fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
fig.text(0.5, 0.04, 'frequency', ha='center', fontsize=15, fontweight='bold')

ax1.barh(baseline_hashtag_vocabulary.keys(), baseline_hashtag_vocabulary.values / baseline_hashtag_vocabulary.values.sum(), 0.25)
ax1.tick_params(labelsize=16)
ax1.set_title("Baseline", fontsize='15', fontweight='bold')

ax2.barh(troll_hashtag_vocabulary.keys(), troll_hashtag_vocabulary.values / troll_hashtag_vocabulary.values.sum(), 0.25)
ax2.tick_params(labelsize=16)
ax2.set_title("Troll", fontsize='15', fontweight='bold')

plt.show()
