"""
Uses the twitter sampling API to sample users, their age, their follower and following counts
"""
import pandas as pd
import requests
import time
import json

# TODO: remember to run the following against the file:
# sed -e 's/[\x00-\x09]//g' sample_with_data.csv > sample_with_data2.csv
# sed -e 's/[\x0B-\x1F]//g' sample_with_data2.csv > sample_with_data3.csv

URL = "https://api.twitter.com/2/tweets/sample/stream?expansions=author_id&user.fields=created_at,public_metrics"
HEADERS = {"Authorization": "Bearer AAAAAAAAAAAAAAAAAAAAAOAKNAEAAAAAcU2NZCJ8R8WO%2FDgVKx00BGm8IO4%3Drc61wrLOGqDeu28c7XYqp8DnYKx0vIQ7tTol7L5xMAUNBj50s0"}
RESULT_FILE = "user_data.csv"

response = requests.get(URL, headers=HEADERS, stream=True)

if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

print(response.status_code)

records = []
processed = 0
for response_line in response.iter_lines():
    if response_line:
        json_response = json.loads(response_line)
        try:
            user = json_response["includes"]["users"][0]
        except KeyError:
            print("Includes not found.. skipping.")
            continue
        records.append({"author": user["id"], "created_at": user["created_at"], "followers": user["public_metrics"]["followers_count"], "following": user["public_metrics"]["following_count"]})

        if len(records) > 1000:
            pd.DataFrame(records).to_csv(RESULT_FILE, index=False, header=False, mode='a')
            processed += len(records)
            records = []
            print(f"Number of users processed: {processed}")

