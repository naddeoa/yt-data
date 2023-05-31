# -*- coding: utf-8 -*-

# Sample Python code for youtube.search.list
# See instructions for running these code samples locally:
# https://developers.google.com/explorer-help/code-samples#python

import os
import json

from pathlib import Path
from pprint import pprint
from typing import Dict, Any
import google_auth_oauthlib.flow
import googleapiclient.discovery  #  ignore: types
import googleapiclient.errors  #  ignore: types
from google.oauth2 import service_account  #  ignore: types
from google.oauth2.credentials import Credentials  #  ignore: types
from typing import TypedDict, Optional
import requests  #  ignore: types
import json
from tqdm import tqdm  #  ignore: types
from terms import search_terms

scopes = ["https://www.googleapis.com/auth/youtube.force-ssl"]


class VideoData(TypedDict):
    title: str
    thumbnail_url: str
    channel_title: str
    description: str
    id: str
    views: Optional[str]


def yoink_info(response_items: dict) -> VideoData:
    return {
        "title": response_items["snippet"]["title"],
        "thumbnail_url": response_items["snippet"]["thumbnails"]["high"]["url"],
        "channel_title": response_items["snippet"]["channelTitle"],
        "description": response_items["snippet"]["description"],
        "id": response_items["id"]["videoId"],
        "views": None,
    }


def save_info(info: Any) -> None:
    id = info["id"]
    img = requests.get(info["thumbnail_url"])
    with open(f"./data/{id}.jpg", "wb") as f:
        f.write(img.content)

    with open(f"./data/{id}.json", "w") as f:
        json.dump(info, f, indent=4)


def init_client() -> Any:
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    credentials_file = "./credentials.json"

    anthony_secret = './secrets/anthony-secrets.json'
    livelock_secret = './secrets/livelockgg-secrets.json'
    thumbs_secret = './secrets/thumbs-project-gg-secrets.json'
    client_secrets_file = livelock_secret

    try:
        # Load the credentials from the file
        credentials = Credentials.from_authorized_user_file(credentials_file)
    except Exception as e:
        # print(e)

        # Get credentials and create an API client
        flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
            client_secrets_file, scopes
        )
        credentials = flow.run_local_server()

        # Save the credentials for future use
        Path(credentials_file).touch()
        with open("./credentials.json", "w") as f:
            f.write(credentials.to_json())

    return googleapiclient.discovery.build(
        api_service_name, api_version, credentials=credentials
    )


def enrich_view_counts(youtube: Any) -> None:
    viewless_metadata: Dict[str, dict] = {}
    for filename in os.listdir("./data"):
        file_path = os.path.join("./data", filename)
        if filename.endswith(".json") and os.path.isfile(file_path):
            with open(file_path, "r") as file:
                json_data = file.read()
                data_dict = json.loads(json_data)
                if "viewCount" not in data_dict or type(data_dict["viewCount"]) == str:
                    viewless_metadata[data_dict["id"]] = data_dict

    print(f"Found {len(viewless_metadata)} videos without view counts.")

    batch_size = 50
    items = list(viewless_metadata.items())

    handled = 0
    for i in tqdm(range(0, len(items), batch_size)):
        batch = items[i : i + batch_size]

        csv_ids = ",".join([id for id, _ in batch])
        request = youtube.videos().list(part="snippet,statistics", id=csv_ids)
        response = request.execute()

        # get statistics.viewCount, update the dict in ids, save the json file again
        for item in response["items"]:
            id = item["id"]
            viewless_metadata[id]["viewCount"] = int(item["statistics"]["viewCount"])
            handled += 1

    # Save everything in viewless_metadata to ./data/{id}.json
    for id, metadata in viewless_metadata.items():
        with open(f"./data/{id}.json", "w") as f:
            json.dump(metadata, f, indent=4)

    print("Updated view counts for", handled, "videos")


def search_videos(youtube: Any) -> None:
    progress = tqdm(search_terms, desc="terms", position=0)
    for term in progress:
        progress.set_description(term)
        request = youtube.search().list(
            part="snippet", maxResults=1000, q=term, type="video", order="viewCount"
        )
        response = request.execute()

        items = [
            yoink_info(i)
            for i in response["items"]
            if i["id"]["kind"] == "youtube#video"
        ]

        for info in tqdm(items, desc="thumbnails", position=1, leave=False):
            save_info(info)

        enrich_view_counts(youtube)


def list_topics(youtube: Any) -> None:
    request = youtube.videoCategories().list(part="snippet", regionCode="US")
    response = request.execute()
    pprint(response)


def main() -> None:
    youtube = init_client()
    enrich_view_counts(youtube)
    search_videos(youtube)


if __name__ == "__main__":
    main()
