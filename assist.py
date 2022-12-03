import pandas as pd
import numpy as np

import lightfm
from lightfm import LightFM
import itertools
from lightfm.data import Dataset
from lightfm import cross_validation

from lightfm.evaluation import precision_at_k as lightfm_prec_at_k
from lightfm.evaluation import recall_at_k as lightfm_recall_at_k
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from scipy.sparse import csr_matrix

#read data
movie = pd.read_csv("datasets\\movies.csv")
rating = pd.read_csv("datasets\\ratings.csv")

#remove movies with no genres
movie = movie[movie["genres"].str.contains("(no genres listed)") == False]

#merge the movie and rating dataframe
data = rating.merge(movie, how="left", on="movieId")

#drop the unneeded timestamp column
data=data.drop(columns=["timestamp"])

#drop rows with na values in any column
data = data.dropna(axis = 0, how ='any')

#list movie genres
movie_genre = [txt.split('|') for txt in data['genres']]
all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))

#creating the dataset
dataset = Dataset()
dataset.fit(users=data['userId'], 
            items=data['movieId'],
           item_features=all_movie_genre)

#build the item features (movie genres)
item_features = dataset.build_item_features(
    (x, y) for x,y in zip(data.movieId, movie_genre))

#build interactions
(interactions, weights) = dataset.build_interactions(data.iloc[:, 0:3].values)

n_users, n_items = interactions.shape

def recommendME(model1,movie,dataset,user_id=None,new_user_feature=None,k=5): 
    nmovie=movie.set_index('movieId')
    user_id_map = dataset.mapping()[0][user_id] # just user_id -1 
    scores = model1.predict(user_id_map, np.arange(n_items),item_features=item_features)
    rank = np.argsort(-scores)
    selected_movies =np.array(list(dataset.mapping()[2].keys()))[rank]
    top_items = nmovie.loc[selected_movies]

    return top_items['title'][:k].values

def train_new():
    #build interactions
    (interactions, weights) = dataset.build_interactions(data.iloc[:, 0:3].values)
    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=0.2,
        random_state=np.random.RandomState(40))
    model1 = LightFM(loss='warp', no_components=20, 
                 learning_rate=0.05,                 
                 random_state=np.random.RandomState(40))
    model1.fit(train_interactions, item_features=item_features, epochs=10)

if __name__=="__main__":
    #create the train/test split (80/20)
    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=0.2,
        random_state=np.random.RandomState(40))

    #create and fit the model
    model = LightFM(loss='warp', no_components=20, 
                 learning_rate=0.05,                 
                 random_state=np.random.RandomState(40))
    model.fit(train_interactions, item_features=item_features, epochs=10)

    #see the evaluation score
    eval_precision_lfm = lightfm_prec_at_k(model, train_interactions, k=10, item_features=item_features).mean()
    eval_recall_lfm = lightfm_recall_at_k(model, train_interactions, k=10, item_features=item_features).mean()
    print('Precision: train %.2f, test %.2f.' % (eval_precision_lfm, eval_recall_lfm))

    lightfm.evaluation.auc_score(model, test_interactions, train_interactions, item_features=item_features).mean()