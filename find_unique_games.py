# Code inspired from:
# -   build_feature_index.py
# -   https://github.com/woctezuma/steam-descriptions/blob/master/find_unique_games.py

import json
import logging
from time import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from build_feature_index import get_features_folder_name
from build_feature_index import get_label_database_filename, convert_label_database, get_frozen_app_ids


def populate_database(pooling=None):
    try:
        print('\n[pooling] {}'.format(pooling))
        label_database = np.load(get_label_database_filename(pooling))
    except FileNotFoundError or OSError:
        if pooling is None:
            raise AssertionError()
        else:
            label_database = convert_label_database(pooling)

    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(label_database)

    # query = label_database
    # num_neighbors = 2

    query = None
    num_neighbors = 1

    start = time()
    dist, matches = knn.kneighbors(X=query, n_neighbors=num_neighbors)
    print('Elapsed time: {:.2f} s'.format(time() - start))

    app_ids = get_frozen_app_ids()

    sim_dict = dict()
    for counter, query_app_id in enumerate(app_ids):
        last_index = num_neighbors-1

        second_best_match = matches[counter][last_index]
        second_best_matched_app_id = app_ids[second_best_match]

        second_best_similarity_score = dist[counter][last_index]

        sim_dict[query_app_id] = dict()
        sim_dict[query_app_id]['app_id'] = second_best_matched_app_id
        sim_dict[query_app_id]['similarity'] = second_best_similarity_score

    with open(get_unique_games_file_name(pooling=pooling), 'w') as f:
        json.dump(sim_dict, f)

    return sim_dict


def get_unique_games_file_name(pooling=None):
    unique_games_file_name = get_features_folder_name() + 'unique_games'

    if pooling is not None:
        unique_games_file_name += '.' + pooling

    unique_games_file_name += '.json'

    return unique_games_file_name


def load_sim_dict(pooling=None):
    with open(get_unique_games_file_name(pooling=pooling), 'r') as f:
        sim_dict = json.load(f)

    return sim_dict


def main(pooling=None,
         similarity_threshold=0.149,
         update_sim_dict=True,
         only_print_banners=False,
         use_markdown=True):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if update_sim_dict:
        sim_dict = populate_database(pooling=pooling)
    else:
        sim_dict = load_sim_dict(pooling=pooling)

    # TODO use similarity_threshold

    return


if __name__ == '__main__':
    main()
