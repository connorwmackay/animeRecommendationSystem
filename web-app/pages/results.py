import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    recommender_labels = ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid System']
    distance_evaluations = [cbf_evaluation['average_precision'], collaborative_evaluation['average_precision'], hybrid_evaluation['average_precision']]

    results_dict = {
        "recommender": recommender_labels,
        "average_precision": distance_evaluations
    }

    results_df = pd.DataFrame(data=results_dict)
    results_df = results_df.rename(columns={"recommender": "Recommender", "average_precision": "Average Precision @ K"})

    fig = px.bar(results_df, x='Recommender', y="Average Precision @ K", title='Comparing the Precision @ K (Average) for Different Recommenders (Normal Users)')
    st.markdown("### Average Precision @ K")
    st.plotly_chart(fig)

    hit_rate_dict = {
        "recommender": recommender_labels,
        "hit_rate": [cbf_evaluation['hit_rate'], collaborative_evaluation['hit_rate'], hybrid_evaluation['hit_rate']]
    }

    st.markdown("### Hit Rate")
    hit_rate_df = pd.DataFrame(data=hit_rate_dict)
    st.table(hit_rate_df)

def show_new_user_results():
    recommender_labels = ['Content-Based Filtering', 'Collaborative Filtering', 'Hybrid System']
    distance_evaluations = [new_cbf_evaluation['average_precision'], new_collaborative_evaluation['average_precision'], new_hybrid_evaluation['average_precision']]

    results_dict = {
        "recommender": recommender_labels,
        "average_precision": distance_evaluations
    }

    results_df = pd.DataFrame(data=results_dict)
    results_df = results_df.rename(columns={"recommender": "Recommender", "average_precision": "Average Precision @ K"})

    fig = px.bar(results_df, x='Recommender', y="Average Precision @ K", title='Comparing the Precision @ K (Average) for Different Recommenders (New Users)')
    st.plotly_chart(fig)

    hit_rate_dict = {
        "recommender": recommender_labels,
        "hit_rate": [new_cbf_evaluation['hit_rate'], new_collaborative_evaluation['hit_rate'], new_hybrid_evaluation['hit_rate']]
    }

    st.markdown("### Hit Rate")
    hit_rate_df = pd.DataFrame(data=hit_rate_dict)
    st.table(hit_rate_df)


user_type_option = st.selectbox(
    "Which users to show results for?", 
    ("Normal Users", "New Users"),
    index=0,
)

if user_type_option == "Normal Users":
    show_normal_user_results()
elif user_type_option == "New Users":
    show_new_user_results()