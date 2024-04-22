print("Started importing recommendation_systems_core...")
from recommendation_systems_core import *
print("Finished importing recommendation_systems_core...")
import pickle
import streamlit as st
import pandas as pd
import numpy as np

print("Started getting recommenders...")

st.set_page_config(layout="wide")

@st.cache_data()
def get_recommender(file_name):
    recommender_file = open(file_name, 'rb')
    recommender = pickle.load(recommender_file)
    recommender_file.close()
    return recommender

cbf_recommender = get_recommender('../data/cbf_recommender')
new_cbf_recommender = get_recommender('../data/new_cbf_recommender')

collaborative_recommender = get_recommender('../data/collaborative_recommender')
new_collaborative_recommender = get_recommender('../data/new_collaborative_recommender')

hybrid_recommender = get_recommender('../data/hybrid_recommender')
new_hybrid_recommender = get_recommender('../data/new_hybrid_recommender')

print("Finished getting recommenders...")

st.markdown("# User Recommendations")
st.sidebar.markdown("# User Recommendations")

container = st.container(border=True)
recommender_option = container.selectbox(
    "Which recommender would you like to use?",
    ("Content-Based Filtering Recommender (Normal Users)", "Content-Based Filtering Recommender (New Users)",
     "Collaborative Filtering Recommender (Normal Users)", "Collaborative Filtering Recommender (New Users)",
     "Hybrid Recommender (Normal Users)", "Hybrid Recommender (New Users)"
     ),
     index=0,
)

def get_random_user(is_normal_user):
    if is_normal_user:
        return np.random.choice(train_df.user_id)
    else:
        return np.random.choice(new_users_train_df.user_id)

def get_user_id_from_text(user_id_input: str, is_normal_user=True):
    if user_id_input != "":
        u_id = int(user_id_input)

        if is_normal_user:
            if len(train_df.loc[train_df.user_id == u_id].user_id > 0):
                return u_id
        
        if not is_normal_user:
            if len(new_users_train_df.loc[new_users_train_df.user_id == u_id].user_id > 0):
                return u_id
    
    return None
   
recommender = cbf_recommender
is_normal_user = True

if recommender_option == "Content-Based Filtering Recommender (Normal Users)":
    is_normal_user = True
    recommender = cbf_recommender
elif recommender_option == "Content-Based Filtering Recommender (New Users)":
    is_normal_user = False
    recommender = new_cbf_recommender
elif recommender_option == "Collaborative Filtering Recommender (Normal Users)":
    is_normal_user = True
    recommender = collaborative_recommender
elif recommender_option == "Collaborative Filtering Recommender (New Users)":
    is_normal_user = False
    recommender = new_collaborative_recommender
elif recommender_option == "Hybrid Recommender (Normal Users)":
    is_normal_user = True
    recommender = hybrid_recommender
elif recommender_option == "Hybrid Recommender (New Users)":
    is_normal_user = False
    recommender = new_hybrid_recommender

num_recommendations_input = container.text_input("Enter Number of Recommendations")

user_id_input = container.text_input("Enter User Id")

user_id = get_random_user(is_normal_user=is_normal_user)
user_id_tmp = get_user_id_from_text(user_id_input, is_normal_user=is_normal_user)

if user_id_tmp:
    container.write("Valid user id")
    user_id = user_id_tmp
elif user_id_input != "":
    container.write(f"{user_id_input} is not a valid user id. A random user id will be used instead.")
    user_id_input = ""

n_recommendations = 5
if num_recommendations_input != "":
    n_recommendations = int(num_recommendations_input)

def show_cbf_results():
    recommendation_results = recommender.recommend_user(user_id, n_recommendations)

    if 'synopsis' in recommendation_results:
        recommendation_results["synopsis"] = recommendation_results["synopsis"].str[:100] + "..."

    if 'combined' in recommendation_results:
        recommendation_results.drop("combined", axis='columns', inplace=True)

    if 'distance' in recommendation_results:
        recommendation_results.drop("distance", axis='columns', inplace=True)

    st.table(recommendation_results)

def show_cf_results():
    recommendation_results = recommender.recommend_user(recommender.cf_trainset.to_inner_uid(user_id), n_recommendations)

    results = recommender.get_user_recommendations_df(recommendation_results)
    results["synopsis"] = results["synopsis"].str[:100] + "..."
    
    st.table(results)

def show_hybrid_results():
    recommendation_results = recommender.recommend_user(user_id, n_recommendations)
    results = recommender.get_user_anime_recommendations_df(recommendation_results)
    results["synopsis"] = results["synopsis"].str[:100] + "..."

    st.table(results)

if user_id_input == "":
    container.text(f"A random user will be picked since no user id was specified")

if container.button("Get Recommendations", type="primary"):
    if is_normal_user:
        st.text(f"Recommendations for User Id: {user_id}")
    else:
        st.text(f"Recommendations for New User Id: {user_id}")

    if recommender_option == "Collaborative Filtering Recommender (New Users)" or recommender_option == "Collaborative Filtering Recommender (Normal Users)":
        show_cf_results()
    elif recommender_option == "Hybrid Recommender (Normal Users)" or recommender_option == "Hybrid Recommender (New Users)":
        show_hybrid_results()
    else: # Content-Based Recommender
        show_cbf_results()