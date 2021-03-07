"""
This script takes the sampled user tweets as input and enriches every tweet record with the number of followers and followings at the time
The number of followings/followers is calculated as an average from x million random sampled users
This approach first gets the users age. Then we calculate the users age at the time of the tweet. At last we look up the average follower/following count for that user age.
"""
import pandas as pd
import requests
from datetime import datetime

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def age_at_tweet(tweet_datetime, author):
    user_age = user_data[user_data["id"] == author]["created_at"]
    user_age_weeks = int((tweet_datetime - user_age).days / 7)

    return user_age_weeks

FOLLOWERS_DATASET = "followers_10wk_avg.csv"
FOLLOWING_DATASET = "following_10wk_avg.csv"
DATASET = "sample_with_data3.csv"
URL = "https://api.twitter.com/2/users"
BATCH_SIZE = 100
HEADERS_DEV =  {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAD80MwEAAAAAYLao%2FyVAFLhPqfayG4zkCCbaOlo%3DmgqWX8HrEL440jXYVKXiRIgZkDGWcAOiILQ3gmjlwZHcVMYJEx"}

data = pd.read_csv(DATASET, header=None, names=["author", "content", "publish_date", "post_type", "retweet", "tweet_id"], dtype={"author": str})
followers_data = pd.read_csv(FOLLOWERS_DATASET)
following_data = pd.read_csv(FOLLOWING_DATASET)
author_ids = data["author"].unique().astype(str)
processed_users = 0
records = []

print(f"author_id's size: {len(author_ids)}")

for b in batch(author_ids, BATCH_SIZE):
    query_params = {"ids": ",".join(b), "user.fields": "created_at"}
    response = requests.request("GET", URL, params=query_params, headers=HEADERS_DEV)
    try:
        records.extend(response.json()["data"])
    except KeyError:
        print(f"Response code: {response.status_code}")
        print(f"Error: {response.json()}")
    processed_users += len(b)
    print(f"Processed users: {processed_users}/{len(author_ids)}")

users_created_at = {u["id"]: u["created_at"] for u in response.json()["data"]}

print("Calculating user creation dates")
data["user_created_at"] = data["author"].apply(lambda x: users_created_at[str(x)])
data["user_created_at"] = pd.to_datetime(data["user_created_at"])

print("Converting publish_date")
data["publish_date"] = pd.to_datetime(data["publish_date"])

print("Calculating age at tweet")
data["age_at_tweet"] = data["publish_date"] - data["user_created_at"]
data["age_at_tweet_weeks"] = data["age_at_tweet"].apply(lambda x: int(x.days / 7))

data["followers"] = data["age_at_tweet_weeks"].apply(lambda x: followers_data["followers"].iloc[x])
data["following"] = data["age_at_tweet_weeks"].apply(lambda x: following_data["following"].iloc[x])

data.drop(columns=["user_created_at", "age_at_tweet", "age_at_tweet_weeks"], inplace=True)
data.to_csv("baseline_dataset.csv")
