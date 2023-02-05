# Code inspired from:
# -   build_feature_index.py
# -   https://github.com/woctezuma/steam-descriptions/blob/master/find_unique_games.py

import json
import logging
from time import time

import numpy as np
import steamspypi
from sklearn.neighbors import NearestNeighbors

from build_feature_index import get_features_folder_name
from build_feature_index import (
    get_label_database_filename,
    convert_label_database,
    get_frozen_app_ids,
)


def load_game_names_from_steamspy():
    data = steamspypi.load()

    game_names = dict()
    for app_id in data.keys():
        game_names[app_id] = data[app_id]['name']

    return game_names


def get_app_name(app_id, game_names=None):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/benchmark_utils.py
    if game_names is None:
        game_names = load_game_names_from_steamspy()

    try:
        app_name = game_names[str(app_id)]
    except KeyError:
        app_name = 'Unknown'

    return app_name


def get_store_url(app_id):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/benchmark_utils.py
    store_url = 'https://store.steampowered.com/app/' + str(app_id)
    return store_url


def get_banner_url(app_id):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/benchmark_utils.py
    banner_url = (
        'https://steamcdn-a.akamaihd.net/steam/apps/' + str(app_id) + '/header.jpg'
    )
    return banner_url


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
    # Caveat: the output 'dist' returned by knn.kneighbors() is the 'cosine distance', not the cosine similarity!
    # Reference: https://en.wikipedia.org/wiki/Cosine_similarity
    dist, matches = knn.kneighbors(X=query, n_neighbors=num_neighbors)
    print('Elapsed time: {:.2f} s'.format(time() - start))

    app_ids = get_frozen_app_ids()

    sim_dict = dict()
    for counter, query_app_id in enumerate(app_ids):
        last_index = num_neighbors - 1

        second_best_match = matches[counter][last_index]
        second_best_matched_app_id = app_ids[second_best_match]

        cosine_distance = dist[counter][last_index]
        second_best_similarity_score = 1.0 - cosine_distance

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


def get_small_banner_url(app_id):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/find_unique_games.py
    small_banner_url = (
        'https://steamcdn-a.akamaihd.net/steam/apps/'
        + str(app_id)
        + '/capsule_sm_120.jpg'
    )
    return small_banner_url


def get_bb_code_linked_image(app_id):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/find_unique_games.py
    bb_code_linked_image = '[URL={}][IMG]{}[/IMG][/URL]'.format(
        get_store_url(app_id),
        get_small_banner_url(app_id),
    )
    return bb_code_linked_image


def print_unique_games(
    sim_dict,
    similarity_threshold,
    game_names,
    only_print_banners=False,
    use_markdown=True,
):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/find_unique_games.py
    # Markdown
    # Reference: https://stackoverflow.com/a/14747656
    image_link_str = '[<img alt="{}" src="{}" width="{}">]({})'
    image_width = 150

    sorted_app_ids = sorted(sim_dict.keys(), key=lambda x: sim_dict[x]['similarity'])

    unique_app_ids = []

    for counter, app_id in enumerate(sorted_app_ids):
        similarity_value = sim_dict[app_id]['similarity']
        if similarity_value <= similarity_threshold:
            unique_app_ids.append(app_id)

            app_name = get_app_name(app_id, game_names=game_names)
            if only_print_banners:
                if use_markdown:
                    # Markdown
                    print(
                        image_link_str.format(
                            app_name,
                            get_banner_url(app_id),
                            image_width,
                            get_store_url(app_id),
                        ),
                    )
                else:
                    # BBCode
                    end_of_entry = ' '  # Either a line break '\n' or a space ' '. Prefer spaces if you post to a forum.
                    print(get_bb_code_linked_image(app_id), end=end_of_entry)
            else:
                print(
                    '{}) similarity = {:.2f} ; appID = {} ({})'.format(
                        counter + 1,
                        similarity_value,
                        app_id,
                        app_name,
                    ),
                )

    return unique_app_ids


def main(
    pooling=None,  # Either None, or 'avg', or 'max'
    num_output=250,  # Allows to automatically define a value for 'similarity_threshold' so that N games are output
    similarity_threshold=None,
    update_sim_dict=False,
    only_print_banners=False,
    use_markdown=True,
):
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
    )

    game_names = load_game_names_from_steamspy()

    if update_sim_dict:
        sim_dict = populate_database(pooling=pooling)
    else:
        sim_dict = load_sim_dict(pooling=pooling)

    if similarity_threshold is None:
        sorted_similarity_values = sorted(
            match['similarity'] for match in sim_dict.values()
        )
        similarity_threshold = sorted_similarity_values[num_output]
        print(
            'Similarity threshold is automatically set to {:.2f}'.format(
                similarity_threshold,
            ),
        )

    unique_app_ids = print_unique_games(
        sim_dict,
        similarity_threshold,
        game_names,
        only_print_banners=only_print_banners,
        use_markdown=use_markdown,
    )

    return


if __name__ == '__main__':
    main()
