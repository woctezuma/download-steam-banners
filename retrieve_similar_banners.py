import json

import imagehash
import numpy as np

from build_search_index import get_search_index_filename
from download_steam_banners import get_app_details


def get_search_index_with_hash_as_str(hashmethod):
    with open(get_search_index_filename(hashmethod), 'r') as f:
        search_index_with_hash_as_str = json.load(f)
    return search_index_with_hash_as_str


def get_hash_search_index(search_index_with_hash_as_str):
    hash_search_index = dict()

    for hash_as_str in search_index_with_hash_as_str:
        hash_value = imagehash.hex_to_hash(hash_as_str)
        hash_search_index[hash_value] = search_index_with_hash_as_str[hash_as_str]

    return hash_search_index


def reverse_search_index(hash_search_index):
    reversed_search_index = dict()

    for hash_value in hash_search_index:
        for app_id in hash_search_index[hash_value]:
            reversed_search_index[app_id] = hash_value

    return reversed_search_index


def retrieve_similar_banners(query_hash, hash_search_index, num_neighbors=3):
    reference_hashes = list(hash_search_index.keys())
    distances = np.array([np.abs(query_hash - reference_hash) for reference_hash in reference_hashes])

    # Reference: https://stackoverflow.com/a/23734295
    ind = np.argpartition(distances, num_neighbors)[:num_neighbors]
    sorted_ind = ind[np.argsort(distances[ind])]

    similar_app_ids = []
    for i in sorted_ind:
        reference_hash = reference_hashes[i]
        reference_app_ids = hash_search_index[reference_hash]
        similar_app_ids.append(reference_app_ids)

    return similar_app_ids


def get_store_url(app_id):
    store_url = 'https://store.steampowered.com/app/' + app_id
    return store_url


def test_hash_search_index(hashmethod):
    search_index_with_hash_as_str = get_search_index_with_hash_as_str(hashmethod)

    hash_search_index = get_hash_search_index(search_index_with_hash_as_str)

    reversed_search_index = reverse_search_index(hash_search_index)

    query_app_ids = ['620', '364470', '504230', '583950', '646570', '863550', '794600']

    for query_app_id in query_app_ids:
        app_details = get_app_details(query_app_id)
        app_name = app_details['name']
        print('\nQuery appID: {} ({})'.format(query_app_id, app_name))

        query_hash = reversed_search_index[query_app_id]
        similar_app_ids = retrieve_similar_banners(query_hash, hash_search_index, num_neighbors=3)

        print('Similar appIDs: {}'.format(similar_app_ids))
        for app_id_group in similar_app_ids:
            for (counter, app_id) in enumerate(app_id_group):
                app_details = get_app_details(app_id)
                app_name = app_details['name']
                print('{}) app: {} ({} @ {})'.format(counter + 1, app_id, app_name, get_store_url(app_id)))

    return


def main():
    for hashmethod in ['ahash', 'phash', 'dhash', 'whash-haar', 'whash-db4']:
        print('\nHash method: {}'.format(hashmethod))
        test_hash_search_index(hashmethod)

    return


if __name__ == '__main__':
    main()
