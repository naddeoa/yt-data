# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os

from pprint import pprint
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from typing import TypedDict
import requests
import json
from tqdm import tqdm

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]


def yoink_info(response_items):
    return {
        "title": response_items["snippet"]["title"],
        "thumbnail_url": response_items["snippet"]["thumbnails"]["high"]["url"],
        "channel_title": response_items["snippet"]["channelTitle"],
        "description": response_items["snippet"]["description"],
        "id": response_items["id"]["videoId"],
    }


def save_info(info):
    id = info["id"]
    img = requests.get(info["thumbnail_url"])
    with open(f"./data/{id}.jpg", "wb") as f:
        f.write(img.content)

    with open(f"./data/{id}.json", "w") as f:
        json.dump(info, f, indent=4)


def main():
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    api_key = "AIzaSyAVtfxaGEbi-kVmMvsYP3v3LxSSy6H02HM"
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

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials
    )

    request = youtube.search().list(part="snippet", maxResults=100, q="gaming")
    response = request.execute()

    items = [
        yoink_info(i) for i in response["items"] if i["id"]["kind"] == "youtube#video"
    ]

    for info in tqdm(items):
        save_info(info)


if __name__ == "__main__":
    main()
