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

def write_collected_anime_to_json_file():
    collected_anime_json = json.dumps(collect_anime_data(), indent=4)

    with open('data/anime.json', "w") as json_file:
        json_file.write(collected_anime_json)

# Uncomment to Collect Data on Every Anime Available on MyAnimeList
#write_collected_anime_to_json_file()