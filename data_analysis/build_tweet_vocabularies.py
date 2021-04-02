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
additional_stopwords = ["rt", "-", "amp", "\|", "&", "�", "its", "it", "u", "im", "https", "httpst", "httpstco", "cant", "you", "thats"]

def preprocess(s):
    # TODO: remove links
    lowercase = s.lower()
    no_stopwords = remove_stopwords(lowercase) # remove stopwords 
    no_punctuations = no_stopwords.translate(table) # remove punctuations
    no_multiple_whitespaces = strip_multiple_whitespaces(no_punctuations) # remove multiple whitespaces
    result = no_multiple_whitespaces.split()
    result = [word for word in result if word not in additional_stopwords]

    return result 

### Import data ###
# data = pd.read_csv("troll_data_2016_english.csv", parse_dates=["date"])
# troll_vocabulary = data["content"].apply(lambda x: preprocess(str(x)))
# troll_vocabulary = troll_vocabulary.explode().value_counts() 

# row_headers_troll = troll_vocabulary.head(30).keys()
# column_headers_troll = ["frequency"]
# rcolors_troll = plt.cm.BuPu(np.full(len(row_headers_troll), 0.1))
# ccolors_troll = plt.cm.BuPu(np.full(len(column_headers_troll), 0.1))
# cell_text_troll = troll_vocabulary.head(30).values.reshape(30,1)

# plt.table(cellText=cell_text_troll.round(decimals=2), rowLabels=row_headers_troll, rowColours=rcolors_troll, rowLoc='right', colColours=ccolors_troll, colLabels=column_headers_troll, loc='center', colWidths=[0.3, 0.3])
# plt.axis('off')
# plt.show()
        
baseline_data = pd.read_csv("baseline_dataset_english.csv", lineterminator='\n', usecols=["content"])["content"]
baseline_vocabulary = baseline_data.apply(lambda x: preprocess(str(x)))
baseline_vocabulary = baseline_vocabulary.explode().value_counts()

row_headers = baseline_vocabulary.head(30).keys()
column_headers = ["frequency"]
rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))
cell_text = baseline_vocabulary.head(30).values.reshape(30,1)

plt.table(cellText=cell_text.round(decimals=2), rowLabels=row_headers, rowColours=rcolors, rowLoc='right', colColours=ccolors, colLabels=column_headers, loc='center', colWidths=[0.3, 0.3])
plt.axis('off')
plt.show()
