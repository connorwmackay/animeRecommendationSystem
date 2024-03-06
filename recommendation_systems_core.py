import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import os

anime_df = pd.read_csv('./data/anime.csv')
train_df = pd.read_csv('data/user_ratings_train.csv')
test_df = pd.read_csv('data/user_ratings_test.csv')
new_users_train_df = pd.read_csv('data/new_user_ratings_train.csv')
new_users_test_df = pd.read_csv('data/new_user_ratings_test.csv')

for user_id in new_users_train_df.user_id.unique():
    num_anime_watched = len(new_users_test_df[new_users_test_df.user_id == user_id])
    if num_anime_watched == 0:
        print(user_id, " watched no anime")
        break

# Return a distance matrix using cosine distance
def cosine_distance_metric(user_profile, anime_vector_df):
    distance_mat = cosine_distances(np.array([user_profile["weighted_vector_avg"].tolist()]).reshape((1, -1)), anime_vector_df).reshape(-1)
    return distance_mat

# Return a distance matrix using euclidean distance
def euclidean_distance_metric(user_profile, anime_vector_df):
    distance_mat = euclidean_distances(np.array([user_profile["weighted_vector_avg"].tolist()]).reshape((1, -1)), anime_vector_df).reshape(-1)
    return distance_mat

# Return a distance matrix using manhattan distance
def manhattan_distance_metric(user_profile, anime_vector_df):
    distance_mat = manhattan_distances(np.array([user_profile["weighted_vector_avg"].tolist()]).reshape((1, -1)), anime_vector_df).reshape(-1)
    return distance_mat

from sklearn.metrics import DistanceMetric

def avg_min_distance_evaluation(anime_watched, predicted_anime, distance_metric='minkowski'):
    min_distances = []
    
    distance = DistanceMetric.get_metric(distance_metric)
    
    # Get the minimum distance for each predicted anime
    for prediction in predicted_anime:
        prediction_distances = distance.pairwise(np.array([prediction]).reshape((1, -1)), anime_watched).reshape(-1)

        if len(prediction_distances) > 0:
            min_distances.append(min(prediction_distances))
        else:
            print(f"Warning. User {user_id} with {len(prediction)} predictions and watched {len(anime_watched)} anime, had zero prediction min distances.")
        
    # Get the average minimum distance of the minimum prediction distances
    average_min_distance = sum(min_distances) / len(min_distances)
    return average_min_distance

import math

def score_recommendations(num_recommendations, recommendations, user_ratings, anime_vector_df, distance_relevant_cutoff = 0.35):
    score = {
        "average_precision": 0,
    }
    
    num_not_scored = 0
    num_scored = 0
    
    unique_user_ids = recommendations.keys()
    for user_id in unique_user_ids:
        num_scored += 1
        predicted_anime = recommendations[user_id]["prediction_vectors"]

        # Get the no. of relevant anime
        user_ratings_inst = user_ratings[user_ratings.user_id == int(user_id)]
        if len(user_ratings_inst) == 0:
            print(f"Error. No user with id {user_id} was found.")
            
        average_rating = round(np.average(user_ratings_inst['score']))
        num_relevant_anime = len(user_ratings_inst[user_ratings_inst.score >= average_rating])

        if num_relevant_anime == 0:
            print("Error. No relavent anime. Average Rating:", average_rating)
            print(f"Min Rating: {min(user_ratings_inst['score'])}, Max Rating: {max(user_ratings_inst['score'])}")

        # Get the no. of relevant recommendations
        num_relevant_recommendations = 0
        for prediction in predicted_anime:
            prediction_distances = cosine_distances(np.array([prediction]).reshape((1, -1)), recommendations[int(user_id)]["actual_anime_vectors"]).reshape(-1)
            if len(prediction_distances) > 0:
                distance = min(prediction_distances)
                if distance <= distance_relevant_cutoff:
                    num_relevant_recommendations += 1
            else:
                print(f"Warning. User {user_id} had no prediction distances")

        score["average_precision"] += num_relevant_recommendations / num_recommendations
    
    score["average_precision"] /= num_scored
    print(f"Num Users Not Scored: {num_not_scored}")
    return score

from sklearn.metrics import mean_squared_error

class CBFRecommender:
    def __init__(self, anime_data: pd.DataFrame, user_ratings_data: pd.DataFrame):
        # Cleanup Anime Data
        self.anime_df = anime_data
        self.anime_df.fillna({"genres": ""}, inplace=True)
        self.anime_df.fillna({"synopsis": ""}, inplace=True)

        self.user_ratings_data = user_ratings_data

    # Uses TF-IDF to Vectorize the Anime DataFrame
    def vectorize_anime_data(self, stop_words='english', max_features=50, max_df=0.5, min_df=0.01):
        vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features, max_df=max_df, min_df=min_df)
        self.anime_df['combined'] =  self.anime_df['genres'] + " " + self.anime_df['synopsis']

        anime_features_vector_matrix = vectorizer.fit_transform(self.anime_df['combined'])

        self.anime_vector_df = pd.DataFrame(data=anime_features_vector_matrix.toarray())
        self.anime_vector_df['anime_id'] = self.anime_df['id']
        self.anime_vector_df = self.anime_vector_df.set_index('anime_id')

        return self.anime_vector_df

    # Creates a profile of the specified user
    def create_user_profile(self, user_id: int):
        # Get only this user's ratings
        users_ratings_df = self.user_ratings_data[self.user_ratings_data.user_id == user_id]

        # Prefer to only use anime they rated higher than their avg. rating
        average_rating = np.average(users_ratings_df['score'])
        selected_user_ratings_df = users_ratings_df[users_ratings_df.score > average_rating]

        # If the user hasn't rated any anime higher than thier avg.,
        # then use the median instead
        if selected_user_ratings_df.empty:
            average_rating = np.average(users_ratings_df['score'])
            selected_user_ratings_df = users_ratings_df[users_ratings_df.score >= average_rating]

        # Get the Anime they rated highly
        user_anime_rated_df = self.anime_df[self.anime_df.id.isin(selected_user_ratings_df['anime_id'])]

        # Get the weighted average of the Vector values for the anime they rated
        user_vector_df = self.anime_vector_df.loc[user_anime_rated_df['id']]
        weighted_vector_avg = user_vector_df.mean()

        return {
            "weighted_vector_avg": weighted_vector_avg,
            "vector_df": user_vector_df
        }

    # Get a DataFrame of How Distant Anime Are to the User's Preferences
    def get_user_anime_distance(self, user_id, distance_metric='cosine'):
        # Find user profile in user profiles
        user_profile = self.create_user_profile(user_id)

        # Calculate the distance matrix of the user's weighted vector instance compared to all instances in the anime vector df
        distance_mat = None
        if distance_metric == 'cosine':
            distance_mat = cosine_distance_metric(user_profile, self.anime_vector_df)
        elif distance_metric == 'euclidean':
            distance_mat = euclidean_distance_metric(user_profile, self.anime_vector_df)
        elif distance_metric == 'manhattan':
            distance_mat = manhattan_distance_metric(user_profile, self.anime_vector_df)
        else:
            print(f'Error: The distance metric is invalid. Unable to create distance matric for user with id {user_id}')

        if distance_mat is None:
            print("Error. Couldn't create distance matrix.")
            return None

        # Convert the matrix into a dataframe
        distance_df = pd.DataFrame(data=distance_mat.tolist())
        distance_df = distance_df.rename(columns={0: "distance"})
        distance_df['id'] = self.anime_df['id']

        # Remove all anime that the user has already watched
        distance_df = distance_df.loc[~distance_df.id.isin(user_profile["vector_df"].index)]

        # The closer to 0 the distance is the more similar the anime is to the weighted vector
        distance_df = distance_df.sort_values(by='distance', ascending=True)

        return distance_df

    # Recommend Anime to the User
    def recommend_user(self, user_id, num_recommendations: int, distance_metric='cosine', add_anime_info: bool = True):
        distance_df = self.get_user_anime_distance(user_id, distance_metric=distance_metric)

        # Get the top recommendations and merge the anime data with the dataframe
        top_similar_anime_df = distance_df.iloc[0:num_recommendations]

        if len(top_similar_anime_df) != num_recommendations:
            print(f"Expected {num_recommendations} recommendations, but got {len(top_similar_anime_df)}")

        # Only merge if we want to the anime information too.
        if add_anime_info:
            top_similar_anime_df = top_similar_anime_df.merge(self.anime_df, on='id', how='inner')

        return top_similar_anime_df

    def get_user_anime_scores(self, user_id, distance_metric='cosine'):
        user_scores_df = self.get_user_anime_distance(user_id, distance_metric=distance_metric)
        user_scores_df = user_scores_df.rename(columns={"distance": "score"})
        return user_scores_df
    
    # Make recommendations for every user
    def make_recommendations(self, testing_df: pd.DataFrame, num_recommendations=20, distance_metric='cosine'):
        unique_user_ids = self.user_ratings_data['user_id'].unique()
        recommendations = {}

        for user_id in unique_user_ids:
            # Get this user's recommendations
            actual_anime_vectors = self.anime_vector_df.loc[self.anime_vector_df.index.isin(testing_df.loc[testing_df.user_id == user_id]['anime_id'])]

            if len(actual_anime_vectors) > 0:
                predictions = self.recommend_user(user_id, num_recommendations, distance_metric=distance_metric, add_anime_info = False)
                predicted_anime_vectors = self.anime_vector_df.loc[self.anime_vector_df.index.isin(predictions['id'])]

                # Add them to the dictionary
                recommendations[user_id] = {"actual_anime_vectors": actual_anime_vectors.to_numpy(),
                                            "predictions": predictions,
                                            "prediction_vectors": predicted_anime_vectors.to_numpy()}
            else:
                print("Warning. Not enought anime vectors.")

        return recommendations
    
    # Test How Effective the Model
    def test_recommendations(self, testing_df: pd.DataFrame, num_recommendations=20, distance_metric='cosine', evaluation_distance_metric='cosine'):
        recommendations = self.make_recommendations(testing_df, num_recommendations=num_recommendations, distance_metric=distance_metric)
        
        # Temp: Compare recommendations IDs to testing IDs
        recommendations_users = recommendations.keys()
        testing_users = testing_df.user_id.unique()
        
        ids_not_in_both = []
        for ruid in recommendations_users:
            not_in_both = True
            for tuid in testing_users:
                if ruid == tuid:
                    not_in_both = False
            
            if not_in_both:
                ids_not_in_both.append(ruid)
        
        print("IDs not in both: ")
        print(ids_not_in_both)
        
        score = score_recommendations(num_recommendations, recommendations, testing_df, self.anime_vector_df)
        
        return score

import os

from surprise import BaselineOnly, Dataset, Reader, accuracy, SVD, SVDpp, NMF
from surprise.model_selection import GridSearchCV, train_test_split

def convert_user_ratings_to_surprise_dataset(ur_df):  
    surprise_user_ratings_df = ur_df[["user_id", "anime_id", "score"]]
    surprise_user_ratings_df = surprise_user_ratings_df.rename(columns={"user_id": "userID", "anime_id": "itemID", "score": "rating"})
    
    cf_ratings_reader = Reader(rating_scale=(0, 10))
    cf_ratings_data = Dataset.load_from_df(surprise_user_ratings_df, cf_ratings_reader)
    return cf_ratings_data

class CollaborativeFilteringRecommender:
    # The algorithm must be Matrix Factorization algorithm supported by surprise
    def __init__(self, ratings_dataset, mf_algorithm=SVD, performGridsearch=False):
        self.cf_trainset = ratings_dataset.build_full_trainset()
        
        # Set the model as the best model found by the grid search
        self.cf_model = None
        
        # Perform a Grid Search to Hyptertune the Parameters
        if performGridsearch:
            param_grid = {"n_factors": [50, 100, 150, 200], "n_epochs": [10, 20, 30], "biased": [True, False]}
            if mf_algorithm == SVD:
                param_grid = {"n_factors": [50, 100, 150, 200], "n_epochs": [20, 25, 30], "lr_all": [0.005, 0.0025], "biased": [True, False]}
            elif mf_algorithm == NMF:
                param_grid = {"n_factors": [15, 30, 60], "n_epochs": [50, 60, 70], "biased": [True, False]}
            
            grid_search = GridSearchCV(mf_algorithm, param_grid, measures=["rmse"], cv=3)
            grid_search.fit(ratings_dataset)
            print(f"Best Score for Grid Search was: {grid_search.best_params['rmse']}")
            print(f"The Best Parameters for Grid Search was: {grid_search.best_score['rmse']}")
            self.cf_model = grid_search.best_estimator["rmse"]
        
        if self.cf_model is None:
            self.cf_model = mf_algorithm()
       
        self.cf_model.fit(self.cf_trainset)
    
    def get_user_anime_scores(self, user_id: str):
        # Get every item that this user hasn't watched.
        item_ids = []
        for item_id in self.cf_trainset.all_items():
            if self.cf_trainset.ur.get(item_id) != None:
                item_ids.append(self.cf_trainset.to_raw_iid(item_id))
        
        # Estimate the rating for each item
        item_predictions = []
        for item_rawid in item_ids:
            prediction = self.cf_model.predict(user_id, item_rawid)
            item_predictions.append(prediction)
    
        user_anime_ids = []
        user_scores = []
        for prediction in item_predictions:
            user_scores.append(prediction.est)
            user_anime_ids.append(int(prediction.iid))
        
        user_scores_dict = {"score": user_scores, "id": user_anime_ids}
        user_scores_df = pd.DataFrame(user_scores_dict)
        
        return user_scores_df
    
    def recommend_user(self, user_innerid, num_recommendations):
        user_recommendations = []
        
        user_id = self.cf_trainset.to_raw_uid(user_innerid)
        
        # Get every item that this user hasn't watched.
        item_ids = []
        for item_id in self.cf_trainset.all_items():
            if self.cf_trainset.ur.get(item_id) != None:
                item_ids.append(self.cf_trainset.to_raw_iid(item_id))
        
        # Estimate the rating for each item
        item_predictions = []
        for item_rawid in item_ids:
            prediction = self.cf_model.predict(user_id, item_rawid)
            item_predictions.append(prediction)
        
        # Recommend the top N items
        item_predictions.sort(key=lambda x: x.est, reverse=True)
        
        for prediction in item_predictions[:num_recommendations]:
            user_recommendations.append(prediction.iid)
        
        return user_recommendations
    
    def get_user_recommendations_df(self, user_recommendations: list):
        return anime_df[anime_df.id.isin(user_recommendations)]
    
    def make_recommendations(self, num_recommendations, test_ratings_df, anime_vector_df):
        recommendations = {}
        
        for inner_userid in self.cf_trainset.all_users():
            user_recommendations = self.recommend_user(inner_userid, num_recommendations)
            user_id = self.cf_trainset.to_raw_uid(inner_userid)
            
            actual_anime_vectors = anime_vector_df.loc[anime_vector_df.index.isin(test_ratings_df.loc[test_ratings_df.user_id == user_id]['anime_id'])]
            predicted_anime_vectors = anime_vector_df.loc[anime_vector_df.index.isin(user_recommendations)]
            if len(predicted_anime_vectors) == 0:
                print("No predicted anime vectors were found.")
            
            recommendations[int(user_id)] = {"actual_anime_vectors": actual_anime_vectors.to_numpy(),
                                            "predictions": user_recommendations,
                                            "prediction_vectors": predicted_anime_vectors.to_numpy()}
        
        return recommendations
    
    def test_recommendations(self, num_recommendations, test_ratings_df, anime_vector_df, evaluation_distance_metric='cosine'):
        recommendations = self.make_recommendations(num_recommendations, test_ratings_df, anime_vector_df)
        score = score_recommendations(num_recommendations, recommendations, test_ratings_df, anime_vector_df)
        
        return score

class HybridRecommender:
    def __init__(self, anime_data: pd.DataFrame, user_ratings_data: pd.DataFrame, cbf_distance_metric='cosine'):
        self.cbf_recommender = CBFRecommender(anime_data, user_ratings_data)
        self.cbf_recommender.vectorize_anime_data(stop_words='english', max_features=50, max_df=0.5, min_df=0.01)
        self.cbf_distance_metric = cbf_distance_metric
        self.anime_data = anime_data
        self.user_ratings_data = user_ratings_data
        
        cf_ratings_data = convert_user_ratings_to_surprise_dataset(user_ratings_data)
        self.cf_recommender = CollaborativeFilteringRecommender(cf_ratings_data)
    
    def get_user_combined_scores(self, user_id):
        # Content-Based Filtering
        cbf_scores_df = self.cbf_recommender.get_user_anime_scores(user_id, distance_metric=self.cbf_distance_metric)
        cbf_scores_df = cbf_scores_df.rename(columns={"score": "cbf_score"})
        
        # Collaborative Filtering
        cf_scores_df = self.cf_recommender.get_user_anime_scores(str(user_id))    
        # Perform Min-Max Normalization
        cf_scores_df["score"] = (cf_scores_df["score"] - cf_scores_df["score"].min()) / (cf_scores_df["score"].max() - cf_scores_df["score"].min())
        cf_scores_df["score"] = (1 - cf_scores_df["score"])
        
        cf_scores_df = cf_scores_df.rename(columns={"score": "cf_score"})
        
        # Combine both into one dataframe
        combined_scores_df = cbf_scores_df.merge(cf_scores_df, on='id', how='inner')
        combined_scores_df['combined_score'] = combined_scores_df['cbf_score'] + combined_scores_df['cf_score']
        combined_scores_df.sort_values(by=['combined_score'], ascending=True, inplace=True)
        
        return combined_scores_df
    
    def recommend_user(self, user_id, num_recommendations):
        user_combined_scores_df = self.get_user_combined_scores(user_id)
        user_top_anime_df = user_combined_scores_df.iloc[:num_recommendations]
        
        return user_top_anime_df
    
    def make_recommendations(self, num_recommendations, test_ratings_df, anime_vector_df):
        recommendations = {}
        unique_user_ids = self.user_ratings_data['user_id'].unique()
        
        for user_id in unique_user_ids:
            user_recommendations = self.recommend_user(user_id, num_recommendations)
            actual_anime_vectors = anime_vector_df.loc[anime_vector_df.index.isin(test_ratings_df.loc[test_ratings_df.user_id == user_id]['anime_id'])]
            predicted_anime_vectors = anime_vector_df.loc[anime_vector_df.index.isin(user_recommendations['id'])]
            
            recommendations[user_id] = {"actual_anime_vectors": actual_anime_vectors.to_numpy(),
                                            "predictions": user_recommendations,
                                            "prediction_vectors": predicted_anime_vectors.to_numpy()}
        
        return recommendations
    
    def test_recommendations(self, num_recommendations, test_user_ratings_data: pd.DataFrame, anime_vector_df, evaluation_distance_metric='cosine'):
        recommendations = self.make_recommendations(num_recommendations, test_user_ratings_data, anime_vector_df)
        score = score_recommendations(num_recommendations, recommendations, test_user_ratings_data, anime_vector_df)
        
        return score