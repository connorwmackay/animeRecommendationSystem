"""
This script collects data on every anime (since you can get 500 per request) and
dumps the data into a JSON file.

Additionally, this script will collect the usernames of a number of users
(20 for every page you choose to request). These usernames can then be used
to request the anime list of each user in the username list.
"""

import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv('CLIENT_ID') # An ID needed to authenticate requests to the API
print(f'Client ID: {CLIENT_ID}')

MY_ANIME_LIST_API_URL = 'https://api.myanimelist.net/v2' # The URL of the API that will be called upon
API_REQUEST_DELAY = 1 # A delay (in seconds) to prevent over-use of the API


# Collect data on every anime and return an array of JSON objects
def collect_anime_data():
    anime_rankings_request_queries = 'limit=500&fields=id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,updated_at,media_type,status,genres,my_list_status,num_episodes,start_season,broadcast,source,average_episode_duration,rating,pictures,background,related_anime,related_manga,recommendations,studios,statistics'

    collected_anime = []
    is_next_page_available = True
    current_url = f'{MY_ANIME_LIST_API_URL}/anime/ranking?{anime_rankings_request_queries}'
    while is_next_page_available:
        anime_rankings_request: requests.Response = None
        try:
            anime_rankings_request = requests.get(current_url, headers={'X-MAL-CLIENT-ID': CLIENT_ID})
            time.sleep(API_REQUEST_DELAY)

            for node in anime_rankings_request.json()['data']:
                collected_anime.append(node['node'])
            
            next_page_url = anime_rankings_request.json()['paging'].get('next')
            if next_page_url is None:
                is_next_page_available = False
            else:
                current_url = next_page_url
                print("Collecting next page of anime...")
        except:
            print(f'Exception occured while attempting to make the request: {anime_rankings_request.json()} from the anime rankings endpoint.')

    print("Finished collecting anime...")
    return {
        "anime": collected_anime
    }

# Roughly 20 usernames per page
def collect_usernames(num_pages=25):
    usernames = []

    for page_num in range(1, num_pages+1):
        users_request: requests.Response = None
        try:
            users_request = requests.get(f'https://api.jikan.moe/v4/users?page={page_num}')
            time.sleep(API_REQUEST_DELAY)
            
            for user in users_request.json()['data']:
                usernames.append(user['username'])
            
            print("Collected next page of usernames...")
        except:
            print(f"Exception occured when trying to perform user request: {users_request.json()}")

    print("Collected all usernames..")

    return {
        "usernames": usernames
    }

def collect_anime_lists(json_file):
    usernames = []
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        usernames = json_data["usernames"]
    print(f'Num usernames found: {len(usernames)}')

    anime_lists = []

    for username in usernames:
        anime_list_request: requests.Response = None
        try:
            anime_list_request = requests.get(f'{MY_ANIME_LIST_API_URL}/users/{username}/animelist?fields=list_status&status=completed&limit=1000', headers={'X-MAL-CLIENT-ID': CLIENT_ID})
            time.sleep(API_REQUEST_DELAY * 2)

            user_anime_list = {"username": username, "anime_list": []}
            anime_list_data = anime_list_request.json().get('data')
            if anime_list_data is not None:
                for node in anime_list_data:
                    user_anime_list["anime_list"].append({"anime": node["node"], "list_status": node["list_status"]})

            while anime_list_request.json()['paging'].get('next') is not None:
                time.sleep(API_REQUEST_DELAY * 2)
                
                anime_list_data = anime_list_request.json().get('data')
                if anime_list_data is not None:
                    for node in anime_list_data:
                        user_anime_list["anime_list"].append({"anime": node["node"], "list_status": node["list_status"]})
                
                # Get Next 1000 Anime if Available
                if anime_list_request.json()['paging'].get('next') is not None:
                    anime_list_request = requests.get(f'{anime_list_request.json()["paging"].get("next")}&fields=list_status', headers={'X-MAL-CLIENT-ID': CLIENT_ID})

            anime_lists.append(user_anime_list)
            print(f'Found {username}\'s anime list. They watched {len(user_anime_list["anime_list"])} anime...')
        except:
            print(f"Exception occured when trying to perform user anime list request with: {anime_list_request.status_code}")

    print("Found every user's anime list...")

    return {
        "anime_lists": anime_lists
    }

def print_num_anime_in_anime_lists(json_file):
    anime_lists = []
    with open(json_file, 'r') as file:
        json_data = json.load(file)
        anime_lists = json_data["anime_lists"]
    
    for ind, anime_list in enumerate(anime_lists):
        print(f'{ind+1}. {anime_list["username"]} watched {len(anime_list["anime_list"])} anime.')

def write_json_to_file(data, filename):
    json_data = json.dumps(data, indent=4)

    with open(filename, "w") as json_file:
        json_file.write(json_data)

# Uncomment to Collect Data on Every Anime Available on MyAnimeList
#write_json_to_file(collect_anime_data(), 'data/anime.json')

# Uncomment to Collect a List of Usernames from the Jikan API
#write_json_to_file(collect_usernames(), 'data/usernames.json')

# Uncomment to Collect a List of User Anime List Data (Requires data/usernames.json file)
#write_json_to_file(collect_anime_lists('data/usernames.json'), 'data/user_anime_lists.json')

# Uncomment to Print out How Many Anime Each User Has Watched (Requires data/user_anime_lists.json)
#print_num_anime_in_anime_lists("data/user_anime_lists.json")