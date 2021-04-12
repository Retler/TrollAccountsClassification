import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
from datetime import datetime
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces

plt.style.use('seaborn')
punctuations = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
digits = '0123456789'
table = str.maketrans("", "", punctuations + digits)
additional_stopwords = ["rt", "-", "amp", "\|", "&", "�", "its", "it", "u", "im", "https", "httpst", "httpstco", "cant", "you", "thats", "youre", "#", "dont", "#a", ""]
HIT_RATE = 0

def is_hashtag(word):
    return word != "" and word[0] == "#"

def is_not_hashtag(word):
    return word != "" and word[0] != "#"

def hit_rate(vocab, words):
    global HIT_RATE
    HIT_RATE += 1
    print(f"Calculating hitrate: {HIT_RATE}")
    if len(words) == 0:
        return 0
    hits = vocab.reindex(words).count() / len(words)
    
    return hits

def preprocess(s, word_check=(lambda x: True)):
    # TODO: remove links
    lowercase = s.lower()
    no_stopwords = remove_stopwords(lowercase) # remove stopwords 
    no_punctuations = no_stopwords.translate(table) # remove punctuations
    no_multiple_whitespaces = strip_multiple_whitespaces(no_punctuations) # remove multiple whitespaces
    result = no_multiple_whitespaces.split()
    result = [word for word in result if (word not in additional_stopwords and word_check(word))]
    
    return result 

### Import data ###
data = pd.read_csv("troll_data_2016_english.csv", usecols=["content", "author"], lineterminator='\n', nrows=500000)
#data["hashtags"] = data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
data["words"] = data["content"].apply(lambda x: preprocess(str(x), lambda x: not is_hashtag(x)))
#troll_hashtag_vocabulary = data["hashtags"].explode().value_counts().sort_values().tail(30)
troll_word_vocabulary = data["words"].explode().value_counts().sort_values().tail(30)

print("1")
baseline_data = pd.read_csv("baseline_data_2016_english.csv", lineterminator='\n', usecols=["author", "content"], nrows=1000000)
# baseline_data["hashtags"] = baseline_data["content"].apply(lambda x: preprocess(str(x), is_hashtag))
print("2")
baseline_data["words"] = baseline_data["content"].apply(lambda x: preprocess(str(x), is_not_hashtag))
print("3")
# baseline_hashtag_vocabulary = baseline_data["hashtags"].explode().value_counts().sort_values().tail(30)
baseline_word_vocabulary = baseline_data["words"].explode().value_counts().sort_values().tail(30)
print("4")
# troll_vs_troll_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
troll_authors_words = data.groupby("author")["words"].apply(lambda x: np.concatenate(x.to_numpy()))
troll_vs_troll_word_hitrate = troll_authors_words.apply(lambda x: hit_rate(troll_word_vocabulary, x))
print("5")
# troll_vs_baseline_hashtag_hitrate = data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(baseline_hashtag_vocabulary, x))
troll_vs_baseline_word_hitrate = troll_authors_words.apply(lambda x: hit_rate(baseline_word_vocabulary, x))

print("6")
# baseline_vs_baseline_hashtag_hitrate = baseline_data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(baseline_hashtag_vocabulary, x))
baseline_authors_words = baseline_data.groupby("author")["words"].apply(lambda x: np.concatenate(x.to_numpy()))
baseline_vs_baseline_word_hitrate = baseline_authors_words.apply(lambda x: hit_rate(baseline_word_vocabulary, x))
print("7")
# baseline_vs_troll_hashtag_hitrate = baseline_data.groupby("author")["hashtags"].sum().apply(lambda x: hit_rate(troll_hashtag_vocabulary, x))
baseline_vs_troll_word_hitrate = baseline_authors_words.apply(lambda x: hit_rate(troll_word_vocabulary, x))

# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# fig.text(0.5, 0.04, 'frequency', ha='center')

# ax1.barh(baseline_word_vocabulary.keys(), baseline_word_vocabulary.values / baseline_word_vocabulary.values.sum(), 0.25)
# ax1.tick_params(labelsize=16)

# ax2.barh(troll_word_vocabulary.keys(), troll_word_vocabulary.values / troll_word_vocabulary.values.sum(), 0.25)
# ax2.tick_params(labelsize=16)

# plt.show()
