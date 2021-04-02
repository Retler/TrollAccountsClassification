import requests
import json
import os
import pandas as pd
import time
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

def auth_header():
    """
    Builds and returns the auth header
    """
    #token = os.environ.get("BEARER_TOKEN")
    token = "AAAAAAAAAAAAAAAAAAAAAD80MwEAAAAAEneViMd4FMLHpwn5K0OczM7mSZE%3DpQVSAq7mWssTkuTPWlSHn0XnvmA0wMqb6zk7fGTpATQ9MWBItM"
    headers = {"Authorization": f"Bearer {token}"}

    return headers

def create_url():
    """
    Returns the full sampling API url
    """
    return "https://api.twitter.com/2/tweets/sample/stream?expansions=author_id"

def sample_users(url, headers, ammount):
    """
    Samples 'ammount' of author_id´s from twitters sample API 
    """
    response = requests.get(url, headers=headers, stream=True)
    author_ids = []
    
    print(response.status_code)
    
    for response_line in response.iter_lines():
        if ammount == 0:
            response.close()
            break
        if response_line:
            json_response = json.loads(response_line)
            try:
                if detect(json_response["data"]["text"]) == "en":
                    author_ids.append(json_response["data"]["author_id"])
                    ammount -= 1
            except LangDetectException:
                print(f"Lang detect error. Skipping this tweet: {json_response['data']['text']}")
                continue


    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    return author_ids

def extract_author_ids(res_json):
    """
    Takes a json response of the twitter sampling API and extracts the author_id's
    """
    return pd.DataFrame(res_json["data"])["author_id"].tolist()

def main():
    headers = auth_header()
    url = create_url()
    sample_size = 250     # How many users to sample at once 
    sample_frequency = 60 # In minutes
    sample_number = 1

    while True:
        print(f"Sampling {sample_size} users. Sample number {sample_number}.")
        authors = sample_users(url, headers, sample_size)

        print("Sampled authors:")
        print(authors)

        sample_file = open('sample.txt', 'a')
        sample_file.write(",".join(authors))
        sample_file.close()

        time.sleep(sample_frequency * 60) # Sleep for 'sample_frequency' minutes
        sample_number += sample_number

if __name__ == "__main__":
    main()
