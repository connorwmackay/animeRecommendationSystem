import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

@st.cache_data()
def get_evaluation(file_name):
    evaluation_file = open(file_name, 'rb')
    evaluation = pickle.load(evaluation_file)
    evaluation_file.close()
    return evaluation

cbf_evaluation = get_evaluation('../data/cbf_evaluation')
new_cbf_evaluation= get_evaluation('../data/new_cbf_evaluation')

collaborative_evaluation = get_evaluation('../data/cf_evaluation')
new_collaborative_evaluation = get_evaluation('../data/new_cf_evaluation')

hybrid_evaluation = get_evaluation('../data/hybrid_evaluation')
new_hybrid_evaluation = get_evaluation('../data/new_hybrid_evaluation')

st.markdown("# Results")
st.sidebar.markdown("# Results")

def show_normal_user_results():
    col1, col2 = st.columns(2)

    recommender_labels = ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid System']
    
    hit_rate_results_dict = {
        "recommender": recommender_labels,
        "5": [cbf_evaluation['5']['hit_rate'], collaborative_evaluation['5']['hit_rate'], hybrid_evaluation['5']['hit_rate']],
        "10": [cbf_evaluation['10']['hit_rate'], collaborative_evaluation['10']['hit_rate'], hybrid_evaluation['10']['hit_rate']],
        "20": [cbf_evaluation['20']['hit_rate'], collaborative_evaluation['20']['hit_rate'], hybrid_evaluation['20']['hit_rate']]
    }

    hit_rate_results_df = pd.DataFrame(data=hit_rate_results_dict)
    hit_rate_results_df = hit_rate_results_df.rename(columns={"recommender": "Recommender", "5": "5 Recommendations", "10": "10 Recommendations", "20": "20 Recommendations"})

    fig1 = px.bar(hit_rate_results_df, x='Recommender', y=["5 Recommendations", "10 Recommendations", "20 Recommendations"], title='Comparing the Hit Rate for Different Recommenders (Normal Users)', barmode='group')
    col1.markdown("### Hit Rate")
    col1.markdown("The hit rate is the ratio of users that had a least one of the anime they rated in the testing set recommended to them. For example, a hit rate of 0.2 means 20% of users had at least one hit.")
    col1.plotly_chart(fig1)

    col1.markdown("#### Table of Hit Rate Results")
    col1.table(hit_rate_results_df)

    mrr_results_dict = {
        "recommender": recommender_labels,
        "5": [cbf_evaluation['5']['mean_reciprocal_rank'], collaborative_evaluation['5']['mean_reciprocal_rank'], hybrid_evaluation['5']['mean_reciprocal_rank']],
        "10": [cbf_evaluation['10']['mean_reciprocal_rank'], collaborative_evaluation['10']['mean_reciprocal_rank'], hybrid_evaluation['10']['mean_reciprocal_rank']],
        "20": [cbf_evaluation['20']['mean_reciprocal_rank'], collaborative_evaluation['20']['mean_reciprocal_rank'], hybrid_evaluation['20']['mean_reciprocal_rank']]
    }

    mrr_results_df = pd.DataFrame(data=mrr_results_dict)
    mrr_results_df = mrr_results_df.rename(columns={"recommender": "Recommender", "5": "5 Recommendations", "10": "10 Recommendations", "20": "20 Recommendations"})

    fig2 = px.bar(mrr_results_df, x='Recommender', y=["5 Recommendations", "10 Recommendations", "20 Recommendations"], title='Comparing the Mean Reciprocal Rank for Different Recommenders (Normal Users)', barmode='group')
    col2.markdown("### Mean Reciprocal Rank")
    col2.markdown("The mean reciprocal rank (MRR) measures the average value for where the first hit was found for every user. For example, if every user had their first hit be the first thing they were recommended then the MRR would be 1, or if it was the second thing then the MRR would be 0.5.")
    col2.plotly_chart(fig2)

    col2.markdown("#### Table of Mean Reciprocal Rank Results")
    col2.table(mrr_results_df)

def show_new_user_results():
    col1, col2 = st.columns(2)

    recommender_labels = ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid System']
    
    hit_rate_results_dict = {
        "recommender": recommender_labels,
        "5": [new_cbf_evaluation['5']['hit_rate'], new_collaborative_evaluation['5']['hit_rate'], new_hybrid_evaluation['5']['hit_rate']],
        "10": [new_cbf_evaluation['10']['hit_rate'], new_collaborative_evaluation['10']['hit_rate'], new_hybrid_evaluation['10']['hit_rate']],
        "20": [new_cbf_evaluation['20']['hit_rate'], new_collaborative_evaluation['20']['hit_rate'], new_hybrid_evaluation['20']['hit_rate']]
    }

    hit_rate_results_df = pd.DataFrame(data=hit_rate_results_dict)
    hit_rate_results_df = hit_rate_results_df.rename(columns={"recommender": "Recommender", "5": "5 Recommendations", "10": "10 Recommendations", "20": "20 Recommendations"})

    fig1 = px.bar(hit_rate_results_df, x='Recommender', y=["5 Recommendations", "10 Recommendations", "20 Recommendations"], title='Comparing the Hit Rate for Different Recommenders (New Users)', barmode='group')
    col1.markdown("### Hit Rate")
    col1.markdown("The hit rate is the ratio of users that had a least one of the anime they rated in the testing set recommended to them. For example, a hit rate of 0.2 means 20% of users had at least one hit.")
    col1.plotly_chart(fig1)

    col1.markdown("#### Table of Hit Rate Results")
    col1.table(hit_rate_results_df)

    mrr_results_dict = {
        "recommender": recommender_labels,
        "5": [new_cbf_evaluation['5']['mean_reciprocal_rank'], new_collaborative_evaluation['5']['mean_reciprocal_rank'], new_hybrid_evaluation['5']['mean_reciprocal_rank']],
        "10": [new_cbf_evaluation['10']['mean_reciprocal_rank'], new_collaborative_evaluation['10']['mean_reciprocal_rank'], new_hybrid_evaluation['10']['mean_reciprocal_rank']],
        "20": [new_cbf_evaluation['20']['mean_reciprocal_rank'], new_collaborative_evaluation['20']['mean_reciprocal_rank'], new_hybrid_evaluation['20']['mean_reciprocal_rank']]
    }

    mrr_results_df = pd.DataFrame(data=mrr_results_dict)
    mrr_results_df = mrr_results_df.rename(columns={"recommender": "Recommender", "5": "5 Recommendations", "10": "10 Recommendations", "20": "20 Recommendations"})

    fig2 = px.bar(mrr_results_df, x='Recommender', y=["5 Recommendations", "10 Recommendations", "20 Recommendations"], title='Comparing the Mean Reciprocal Rank for Different Recommenders (New Users)', barmode='group')
    col2.markdown("### Mean Reciprocal Rank")
    col2.markdown("The mean reciprocal rank (MRR) measures the average value for where the first hit was found for every user. For example, if every user had their first hit be the first thing they were recommended then the MRR would be 1, or if it was the second thing then the MRR would be 0.5.")
    col2.plotly_chart(fig2)

    col2.markdown("#### Table of Mean Reciprocal Rank Results")
    col2.table(mrr_results_df)

user_type_option = st.selectbox(
    "Which users to show results for?", 
    ("Normal Users", "New Users"),
    index=0,
)

if user_type_option == "Normal Users":
    show_normal_user_results()
elif user_type_option == "New Users":
    show_new_user_results()