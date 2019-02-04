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


def get_search_index_filename(hashmethod=None):
    if hashmethod is None:
        hashmethod_str = ''
    else:
        hashmethod_str = '_' + hashmethod
    search_index_filename = 'search_index' + hashmethod_str + '.json'
    return search_index_filename


def app_id_to_image_filename(app_id):
    image_filename = get_data_folder() + app_id + get_file_extension()
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
        hashfunc = lambda img: imagehash.whash(img, mode='db4')

    image_filenames = Path(get_data_folder()).glob('*' + get_file_extension())

    app_ids = [banner.name.strip(get_file_extension()) for banner in image_filenames]

    num_games = len(app_ids)

    start = time()

    try:
        with open(get_search_index_filename(hashmethod), 'r') as f:
            search_index = json.load(f)
    except FileNotFoundError:
        search_index = dict()

    for (counter, app_id) in enumerate(sorted(app_ids, key=int)):

        if (counter % 1000) == 0:
            print('[{}/{}] appID = {}'.format(counter, num_games, app_id))
            print('Elapsed time: {:.2f} s'.format(time() - start))
            start = time()

        image_filename = app_id_to_image_filename(app_id)
        image = Image.open(image_filename)
        hash = hashfunc(image)
        try:
            search_index[hash].add(app_id)
        except KeyError:
            search_index[hash] = set()
            search_index[hash].add(app_id)

    with open(get_search_index_filename(hashmethod), 'w') as f:
        json.dump(search_index, f)

    return search_index


if __name__ == '__main__':
    for hashmethod in ['ahash', 'phash', 'dhash', 'whash-haar', 'whash-db4']:
        print('Hash method: {}'.format(hashmethod))
        search_index = build_search_index(hashmethod)
