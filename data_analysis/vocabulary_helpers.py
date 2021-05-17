from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces

punctuations = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
digits = '0123456789'
table = str.maketrans("", "", punctuations + digits)
additional_stopwords = ["rt", "-", "amp", "\|", "&", "�", "its", "it", "u", "im", "cant", "you", "thats", "youre", "#", "dont", "#a", ""]
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
    result = [word for word in result if (word not in additional_stopwords and word_check(word) and "https" not in word)]
    
    return result 
