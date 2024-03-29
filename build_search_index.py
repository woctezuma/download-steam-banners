import json
from pathlib import Path
from time import time

import imagehash
from PIL import Image


def get_data_folder():
    data_folder = 'data/'
    return data_folder


def get_file_extension():
    file_extension = '.jpg'
    return file_extension


def get_hash_folder():
    hash_folder = 'hash/'
    return hash_folder


def get_search_index_filename(hashmethod=None):
    hashmethod_str = '' if hashmethod is None else '_' + hashmethod
    search_index_filename = (
        get_hash_folder() + 'search_index' + hashmethod_str + '.json'
    )
    return search_index_filename


def list_app_ids(data_folder=None):
    if data_folder is None:
        data_folder = get_data_folder()

    image_filenames = Path(data_folder).glob('*' + get_file_extension())

    app_ids = [banner.name.strip(get_file_extension()) for banner in image_filenames]

    app_ids = sorted(app_ids, key=int)

    return app_ids


def app_id_to_image_filename(app_id, data_folder=None):
    if data_folder is None:
        data_folder = get_data_folder()

    Path(data_folder).mkdir(exist_ok=True)
    image_filename = data_folder + str(app_id) + get_file_extension()
    return image_filename


def build_search_index(hashmethod=None):
    if hashmethod == 'ahash':  # Average hash
        hashfunc = imagehash.average_hash
    elif hashmethod == 'phash':  # Perceptual hash
        hashfunc = imagehash.phash
    elif hashmethod == 'dhash':  # Difference hash
        hashfunc = imagehash.dhash
    elif hashmethod == 'whash-haar':  # Haar wavelet hash
        hashfunc = imagehash.whash
    else:  # Daubechies wavelet hash

        def hashfunc(img):
            return imagehash.whash(img, mode='db4')

    app_ids = list_app_ids()

    num_games = len(app_ids)

    start = time()

    try:
        with open(get_search_index_filename(hashmethod)) as f:
            search_index = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        search_index = {}

    for counter, app_id in enumerate(sorted(app_ids, key=int)):
        if (counter % 1000) == 0:
            print(f'[{counter}/{num_games}] appID = {app_id}')
            print(f'Elapsed time: {time() - start:.2f} s')
            start = time()

        image_filename = app_id_to_image_filename(app_id)
        image = Image.open(image_filename)
        hash_value = hashfunc(image)
        hash_as_str = str(
            hash_value,
        )  # NB: To convert back, use imagehash.hex_to_hash()
        try:
            if app_id not in search_index[hash_as_str]:
                search_index[hash_as_str].append(app_id)
        except KeyError:
            search_index[hash_as_str] = [app_id]

    with open(get_search_index_filename(hashmethod), 'w') as f:
        json.dump(search_index, f)

    return search_index


if __name__ == '__main__':
    for hashmethod in ['ahash', 'phash', 'dhash', 'whash-haar', 'whash-db4']:
        print(f'Hash method: {hashmethod}')
        search_index = build_search_index(hashmethod)
