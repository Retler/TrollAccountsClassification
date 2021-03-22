import pandas as pd
import numpy as np
from datetime import datetime
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

USERS_PARSED = 0

def calculate_user_language(authors_tweets):
    global USERS_PARSED
    english_tweets = [] 

    for tweet in authors_tweets:
        try:
            english_tweets.append(detect(str(tweet)) == "en")
        except LangDetectException:
            english_tweets.append(False)

    USERS_PARSED += 1
    print(f"Parsed {USERS_PARSED}/{users_total} users")
    print(f"sum english tweets: {sum(english_tweets)}, len english tweets: {len(english_tweets)}")

    return sum(english_tweets) / len(english_tweets) >= 0.5 # user language is english if >= 50% of the tweets are english

baseline_data = pd.read_csv("baseline_dataset.csv", parse_dates=["publish_date"])
users_total = len(baseline_data["author"].unique())
english_users = baseline_data.groupby("author", group_keys=False).apply(lambda x: x.sample(n=10, replace=True)).groupby("author")["content"].apply(calculate_user_language)
baseline_data[baseline_data["author"].map(english_users)].to_csv("baseline_dataset_english.csv")