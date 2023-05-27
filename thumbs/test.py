# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os
import json

from pprint import pprint
from typing import Dict
import google_auth_oauthlib.flow
import googleapiclient.discovery  #  ignore: types
import googleapiclient.errors  #  ignore: types
from google.oauth2 import service_account  #  ignore: types
from google.oauth2.credentials import Credentials  #  ignore: types
from typing import TypedDict
import requests  #  ignore: types
import json
from tqdm import tqdm  #  ignore: types
from terms import search_terms

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]


def yoink_info(response_items):
    return {
        "title": response_items["snippet"]["title"],
        "thumbnail_url": response_items["snippet"]["thumbnails"]["high"]["url"],
        "channel_title": response_items["snippet"]["channelTitle"],
        "description": response_items["snippet"]["description"],
        "id": response_items["id"]["videoId"],
        # "views": response_items["statistics"]["viewCount"],
        # "likeCount": response_items["statistics"]["likeCount"],
        # "commentCount": response_items["statistics"]["commentCount"],
    }


def save_info(info):
    id = info["id"]
    img = requests.get(info["thumbnail_url"])
    with open(f"./data/{id}.jpg", "wb") as f:
        f.write(img.content)

    with open(f"./data/{id}.json", "w") as f:
        json.dump(info, f, indent=4)

def init_client():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    client_secrets_file = "./secrets.json"

    try:
        # Load the credentials from the file
        credentials = Credentials.from_authorized_user_file("./credentials.json")
    except Exception as e:
        print(e)

        # Get credentials and create an API client
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            client_secrets_file, scopes
        )
        credentials = flow.run_local_server()

        # Save the credentials for future use
        with open("credentials.json", "w") as f:
            f.write(credentials.to_json())

    return googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials
    )

def apply_view_counts(youtube):
    viewless_metadata: Dict[str, dict] = {}
    for filename in os.listdir('./data'):
        file_path = os.path.join('./data', filename)
        if filename.endswith('.json') and os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                json_data = file.read()
                data_dict = json.loads(json_data)
                if 'viewCount' not in data_dict or  type(data_dict['viewCount']) == str:
                    viewless_metadata[data_dict['id']] = data_dict

    print(f'Found {len(viewless_metadata)} videos without view counts.')

    batch_size = 50
    items = list(viewless_metadata.items())

    handled = 0
    for i in tqdm(range(0, len(items), batch_size)):
        batch = items[i:i+batch_size]

        csv_ids = ','.join([id for id, _ in batch])
        request = youtube.videos().list(
            part="snippet,statistics",
            id=csv_ids
        )
        response = request.execute()

        # get statistics.viewCount, update the dict in ids, save the json file again
        for item in response['items']:
            id = item['id']
            viewless_metadata[id]['viewCount'] = int(item['statistics']['viewCount'])
            handled += 1
            # These aren't always here
            # viewless_metadata[id]['likeCount'] = item['statistics']['likeCount'] 
            # viewless_metadata[id]['commentCount'] = item['statistics']['commentCount']

        
    # Save everything in viewless_metadata to ./data/{id}.json
    for id, metadata in viewless_metadata.items():
        with open(f"./data/{id}.json", "w") as f:
            json.dump(metadata, f, indent=4)

    print('Updated view counts for', handled, 'videos')


def search_videos(youtube):
    progress = tqdm(search_terms, desc='terms', position=0)
    for term in progress:
        progress.set_description(term)
        request = youtube.search().list(part="snippet", maxResults=1000, q=term, type='video', order='viewCount')
        response = request.execute()

        items = [
            yoink_info(i) for i in response["items"] if i["id"]["kind"] == "youtube#video"
        ]

        for info in tqdm(items, desc='thumbnails', position=1, leave=False):
            save_info(info)


def list_topics(youtube):
    request = youtube.videoCategories().list(
        part="snippet",
        regionCode="US"
    )
    response = request.execute()
    pprint(response)


def main():
    youtube = init_client()
    # search_videos(youtube)
    apply_view_counts(youtube)

if __name__ == "__main__":
    main()
