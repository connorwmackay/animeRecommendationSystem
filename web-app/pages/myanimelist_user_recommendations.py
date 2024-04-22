import pickle
import streamlit as st
import pandas as pd
import numpy as np

from dotenv import load_dotenv

load_dotenv("../.env")

print("Started importing recommendation_systems_core...")
from recommendation_systems_core import *
print("Finished importing recommendation_systems_core...")

st.set_page_config(layout="wide")

print("Started getting recommenders...")

@st.cache_data()
def get_recommender(file_name):
    recommender_file = open(file_name, 'rb')
    recommender = pickle.load(recommender_file)
    recommender_file.close()
    return recommender

cbf_recommender = get_recommender('../data/cbf_recommender')
collaborative_recommender = get_recommender('../data/collaborative_recommender')
hybrid_recommender = get_recommender('../data/hybrid_recommender')

print("Finished getting recommenders...")

st.markdown("# Get Recommendations For MyAnimeList Accounts")
st.sidebar.markdown("# Recommendations For MyAnimeList Accounts")

# Recommender Selector
container = st.container(border=True)
recommender_option = container.selectbox(
    "Which recommender would you like to use?",
    ("Content-Based Filtering Recommender",
     "Collaborative Filtering Recommender",
     "Hybrid Recommender",
     ),
     index=0,
)

recommender = cbf_recommender
user_id = -1

# Username Text Input
username = container.text_input("Enter MyAnimeList Username")

# Num. Of Recommendations Input
num_recommendations_input = container.text_input("Enter Number of Recommendations")

n_recommendations = 5
if num_recommendations_input != "":
    n_recommendations = int(num_recommendations_input)

if recommender_option == "Content-Based Filtering Recommender":
    recommender = cbf_recommender
elif recommender_option == "Collaborative Filtering Recommender":
    recommender = collaborative_recommender
elif recommender_option == "Hybrid Recommender":
    recommender = hybrid_recommender

def show_cbf_results():
    user_id = recommender.add_new_user_by_mal_username(username)
    recommendation_results = recommender.recommend_user(user_id, n_recommendations)

    if 'synopsis' in recommendation_results:
        recommendation_results["synopsis"] = recommendation_results["synopsis"].str[:100] + "..."

    if 'combined' in recommendation_results:
        recommendation_results.drop("combined", axis='columns', inplace=True)

    if 'distance' in recommendation_results:
        recommendation_results.drop("distance", axis='columns', inplace=True)

    st.table(recommendation_results)

def show_cf_results():
    user_id = recommender.add_new_user_by_mal_username(username)
    recommendation_results = recommender.recommend_user(recommender.cf_trainset.to_inner_uid(user_id), n_recommendations)

    results = recommender.get_user_recommendations_df(recommendation_results)
    results["synopsis"] = results["synopsis"].str[:100] + "..."
    
    st.table(results)

def show_hybrid_results():
    user_id = recommender.add_new_user_by_mal_username(username)
    recommendation_results = recommender.recommend_user(user_id, n_recommendations)
    results = recommender.get_user_anime_recommendations_df(recommendation_results)
    results["synopsis"] = results["synopsis"].str[:100] + "..."

    st.table(results)

if container.button("Get Recommendations", type="primary"):
    print(f"Using the username {username}")

    if recommender_option == "Content-Based Filtering Recommender":
        show_cbf_results()
    elif recommender_option == "Collaborative Filtering Recommender":
        show_cf_results()
    elif recommender_option == "Hybrid Recommender":
        show_hybrid_results()
    
    username = ""