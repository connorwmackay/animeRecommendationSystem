"""
This script will convert the JSON files I
create using the apiDataCollector script and
turn them into CSV files, usable with Pandas.
"""

import requests
import json
import csv
import pandas as pd
import numpy as np

def convert_anime_json_to_csv():
    anime_list = []
    
    # Get the list of anime from the JSON file
    with open('data/anime.json', 'r') as anime_json_file:
        anime_json = json.load(anime_json_file)
        anime_list = anime_json["anime"]
    
    # Create and write to a CSV file
    with open('data/anime.csv', 'w', newline='\n', encoding="utf-8") as csv_file:
        anime_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write Header Row
        anime_writer.writerow(['id', 'title', 'start_date', 'end_date', 'synopsis', 'score', 'rank', 
                              'popularity', 'num_list_users', 'num_scoring_users', 'media_type', 'status', 'genres', 
                              'num_episodes', 'start_season', 'broadcast_day', 'broadcast_time', 'source', 'rating',
                              'studios'])
        
        # Write Each Item As a Row
        for anime in anime_list:
            # Convert genre list into friendly format
            anime_genres = ''
            if anime.get('genres') is not None:
                for ind, genre in enumerate(anime.get('genres')):
                    anime_genres += genre["name"]
                    if ind < len(anime.get('genres'))-1:
                        anime_genres += ','

            # Convert studio list into friendly format
            anime_studios = ''
            if anime.get('studios') is not None:
                for ind, studio in enumerate(anime.get('studios')):
                    anime_studios += studio['name']
                    if ind < len(anime.get('studios'))-1:
                        anime_studios += ','

            # Extra Code for if Broadcast is Missing
            broadcast_day = ''
            broadcast_time = ''
            if anime.get('broadcast') is not None:
                if anime['broadcast'].get('day_of_the_week') is not None:
                    broadcast_day = anime['broadcast']['day_of_the_week']
                
                if anime['broadcast'].get('start_time') is not None:
                    broadcast_time = anime['broadcast']['start_time']

            # Extra code for if Start Season is Missing
            start_season = ''
            if anime.get('start_season') is not None:
                start_season = anime['start_season']['season']

            # Write Anime Row
            anime_writer.writerow([anime.get('id'), anime.get('title'), anime.get('start_date'), anime.get('end_date'), anime.get('synopsis'),
                                  anime.get('mean'), anime.get('rank'), anime.get('popularity'), anime.get('num_list_users'), 
                                  anime.get('num_scoring_users'), anime.get('media_type'), anime.get('status'), anime_genres,
                                  anime.get('num_episodes'), start_season, broadcast_day,
                                  broadcast_time, anime.get('source'), anime.get('rating'), anime_studios])

# Uncomment to Generate the CSV file (Requires 'data/anime.json')
#convert_anime_json_to_csv()

# Uncomment for to Check Basic CSV Info
#anime_df = pd.read_csv('data/anime.csv')
#print(anime_df.head())
#print(anime_df.describe())