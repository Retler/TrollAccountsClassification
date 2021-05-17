import csv
import requests
import os
import time
import pandas as pd
import random

HEADERS = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAOAKNAEAAAAAtQ9RmQQkKCa2a7aHC%2Bps6CUElIQ%3DjR22nKwAU6OIxlWFSKT9KckWzrslAuDwwgkyRbGJyFaEGptj47"}
URL = "https://api.twitter.com/2/tweets/search/all"
LIVENESS_URL = "https://europe-west3-es-staging-276908.cloudfunctions.net/liveness-probe"

def log(message):
    print(message)

def read_author_ids() -> [str]:
    with open('celeb_ids.txt') as f:
        author_ids = f.read().split(',')

    return author_ids

def get_tweet_type(referenced_tweets):
    if len(referenced_tweets) == 0:
        return float('nan')

    if referenced_tweets[0]["type"] == "retweeted":
        return "RETWEET"

    if referenced_tweets[0]["type"] == "quoted":
        return "QUOTE_TWEET"
    
    return float('nan')

def data_to_record(data, author_id):
    records = []
    if data["meta"]["result_count"] > 0:
        created_at_first = data["data"][0]["created_at"]
        log(f"Processing tweets dated earlier than {created_at_first}")
        
        for dat in data["data"]:
            try:
                tweet_type = get_tweet_type(dat["referenced_tweets"])
            except KeyError:
                tweet_type = float('nan')
            retweet = int(tweet_type == "RETWEET")
            content = dat["text"].replace('\n','')
            record = {"author": author_id, "content": content, "publish_date": dat["created_at"], "post_type": tweet_type, "retweet": retweet, "tweet_id": dat["id"]}
            records.append(record)
    else:
        log("No tweets for this user in the given time period. Skipping..")
        
    return records
    
def get_authors_tweets(author_id, start_time, end_time):
    log(f"Processing author {author_id}")
    query_params = {'query': f'(from:{author_id})', 'start_time': start_time, 'end_time': end_time, 'max_results':'500', 'tweet.fields':'created_at,referenced_tweets'}
    response = requests.request("GET", URL, headers=HEADERS, params=query_params)

    if response.status_code != 200:
        print(f"Got status code {response.status_code} with response {response.text}! Skipping author {author_id}")
        return []
        
        
        
    result = response.json()
    records = data_to_record(result, author_id)
    pagenum = 1

    while True:
        try:
            time.sleep(3.5)
            log(f"Trying to parse page number {pagenum}")
            query_params["next_token"] = result["meta"]["next_token"]
            response = requests.request("GET", URL, headers=HEADERS, params=query_params)

            if response.status_code == 429:
                log("Reached requests limit! Re-trying in 1 minute..")
                time.sleep(60)
                continue
            if response.status_code == 503:
                log("Service unavailable, waiting 10 seconds.")
                time.sleep(10)
                continue
            if response.status_code != 200:
                log("Response error recieved:")
                log(response.headers)
                log(response.text)
                raise Exception(response.status_code, response.text)
            
            result = response.json()
            new_records = data_to_record(result, author_id)
            records.extend(new_records)
            pagenum += 1
        except KeyError:
            log(f"No more pages left. Parsed pages: {pagenum}")
            break
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as e:
            log(f"Failed to establish a connection! Retrying in 10 seconds")
            time.sleep(10)
            continue
    
    return records

def main():
    author_ids = read_author_ids()
    
    start_time = "2016-01-01T00:00:00Z"
    end_time = "2016-12-31T23:59:59Z"
    result_file = "celeb_tweets.csv"
    
    for author_id in author_ids:
        result = get_authors_tweets(author_id, start_time, end_time)
        log(f"Parsed {len(result)} records for author {author_id} for {start_time} to {end_time}")
        log(f"Appending records to {result_file}")
        if len(result) > 0:
            keys = result[0].keys()
            with open(result_file, 'a', newline='')  as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writerows(result)
            
if __name__ == "__main__":
    main()
