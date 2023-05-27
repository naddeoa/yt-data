# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os

from pprint import pprint
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
        "views": response_items["statistics"]["viewCount"],
        "likeCount": response_items["statistics"]["likeCount"],
        "commentCount": response_items["statistics"]["commentCount"],
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

def main():
    youtube = init_client()

    progress = tqdm(search_terms, desc='terms', position=0)
    for term in progress:
        progress.set_description(term)
        request = youtube.search().list(part="snippet,statistics", maxResults=1000, q=term)
        response = request.execute()

        items = [
            yoink_info(i) for i in response["items"] if i["id"]["kind"] == "youtube#video"
        ]

        for info in tqdm(items, desc='thumbnails', position=1, leave=False):
            save_info(info)


if __name__ == "__main__":
    main()
