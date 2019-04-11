import asyncio
from collections import Counter
from pathlib import Path
from time import time

import aiofiles
import aiohttp
import cv2 as cv
import numpy as np
import steamspypi
from keras.preprocessing.image import load_img
from sklearn.neighbors import NearestNeighbors

from build_feature_index import get_descriptor_database_filename, get_descriptor_img_id_filename, convert_label_database
from build_feature_index import get_label_database_filename, load_keras_model, label_image, get_frozen_app_ids
from build_search_index import app_id_to_image_filename, list_app_ids
from download_steam_banners import get_app_details
from retrieve_similar_banners import get_store_url


async def download_steam_banner_again(app_id, banner_file_name):
    async with aiohttp.ClientSession() as session:
        if not Path(banner_file_name).exists():
            banner_url = 'https://steamcdn-a.akamaihd.net/steam/apps/' + str(app_id) + '/header.jpg'

            # Reference: https://stackoverflow.com/a/51745925
            async with session.get(banner_url) as resp:
                if resp.status == 200:
                    f = await aiofiles.open(banner_file_name, mode='wb')
                    await f.write(await resp.read())
                    await f.close()
                    print('Banner downloaded to {} for appID {}.'.format(banner_file_name, app_id))
                else:
                    print('Banner for appID {} could not be downloaded.'.format(app_id))

    return


def retrieve_similar_features(query_app_id, descriptor_database=None, descriptor_img_id=None,
                              label_database=None, keras_model=None, target_model_size=None, pooling=None, knn=None,
                              data_folder=None,
                              images_are_store_banners=True):
    image_filename = app_id_to_image_filename(query_app_id, data_folder=data_folder)

    if keras_model is not None:

        if images_are_store_banners and not Path(image_filename).exists():
            print('File {} not found: appID {} likely unavailable in this region.'.format(image_filename, query_app_id))
            loop = asyncio.get_event_loop()
            loop.run_until_complete(download_steam_banner_again(query_app_id, image_filename))

        image = load_img(image_filename, target_size=target_model_size)
        query_des = label_image(image, keras_model)  # runtime: 1 second

        query_des = query_des.flatten()

        # For FLANN, the query and the database should have the same dtype 'float32'.
        query_des = query_des.astype('float32')

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        query_img = cv.imread(image_filename, cv.IMREAD_COLOR)

        # Initiate ORB detector
        orb = cv.ORB_create()

        # find the keypoints and descriptors with ORB
        _, query_des = orb.detectAndCompute(query_img, None)

        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2

    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    if keras_model is not None:
        if label_database is None:
            try:
                label_database = np.load(get_label_database_filename(pooling))
            except OSError:
                label_database = convert_label_database(pooling)

    else:
        if descriptor_database is None:
            descriptor_database = np.load(get_descriptor_database_filename())

        if descriptor_img_id is None:
            descriptor_img_id = np.load(get_descriptor_img_id_filename())

    try:
        app_ids = get_frozen_app_ids()

    except FileNotFoundError:
        print('Assumption: no new banner was downloaded since the last feature pre-computation.')
        app_ids = list_app_ids(data_folder=data_folder)

    if keras_model is not None:
        trimmed_descriptor_database = label_database.copy()
        # Trimming the database is optional here.
        optional_trimming = False
        if optional_trimming:
            try:
                row_no = app_ids.index(query_app_id)
            except ValueError:
                print('AppID {} not part of frozen appIDs: it is likely unavailable in my region.'.format(query_app_id))
                row_no = None

            if row_no is not None:
                trimmed_descriptor_database[row_no, :] = 0

        # For FLANN, the query and the database should have the same dtype 'float32'.
        trimmed_descriptor_database = trimmed_descriptor_database.astype('float32')

        num_neighbors = 10
    else:
        # Remove the features of the query from the database.
        # NB: Trimming the database is MANDATORY because we only keep THE best match for each feature (if the ratio
        # 1stNN/2ndNN<0.7), and then count the number of occurrences of each appID. Otherwise, we would retrieve the
        # query, and no other appID.
        ind, _ = np.where(descriptor_img_id == app_ids.index(query_app_id))
        trimmed_descriptor_database = descriptor_database.copy()
        trimmed_descriptor_database[ind] = 0

        num_neighbors = 2

    start = time()
    if knn is None:
        # FLANN with L2 distance
        matches = flann.knnMatch(query_des, trimmed_descriptor_database, k=num_neighbors)
    else:
        # Sci-Kit Learn with cosine similarity. Reshape data as it contains a single sample.
        _, matches = knn.kneighbors(query_des.reshape(1, -1), n_neighbors=num_neighbors)
    print('Elapsed time: {:.2f} s'.format(time() - start))

    if keras_model is not None:
        # When we use the Keras model, a Steam banner is represented by only ONE feature, hence the use of 'matches[0]'.
        try:
            # FLANN
            reference_app_id_counter = [app_ids[element.trainIdx] for element in matches[0]]
        except AttributeError:
            # Sci-Kit Learn
            reference_app_id_counter = [app_ids[element] for element in matches[0]]
    else:
        # store all the good matches as per Lowe's ratio test.
        good_matches = []
        for t in matches:
            try:
                if t[0].distance < 0.7 * t[1].distance:
                    good_matches.append(t[0])
            except IndexError:
                pass

        good_img_ids = [int(descriptor_img_id[good_match.trainIdx]) for good_match in good_matches]

        good_app_ids = [app_ids[img_id] for img_id in good_img_ids]

        reference_app_id_counter = Counter(good_app_ids)

    return reference_app_id_counter


def print_ranking(query_app_id, reference_app_id_counter, num_elements_displayed=10, only_print_banners=False,
                  use_markdown_syntax=True):
    try:
        app_details = get_app_details(query_app_id)
    except FileNotFoundError:
        print('App details for appID {} not found: appID is likely unavailable in this region'.format(query_app_id))
        app_details = None

    if app_details is not None:
        app_name = app_details['name']
    else:
        app_name = 'Unknown'

    if use_markdown_syntax:
        # Markdown
        print('\nQuery appID: {} ([{}]({}))\n'.format(query_app_id, app_name, get_store_url(query_app_id)))
    else:
        # BBCode
        print('\nQuery appID: {} ([url={}]{}[/url])\n'.format(query_app_id, get_store_url(query_app_id), app_name))

    for rank, app_id in enumerate(reference_app_id_counter):
        app_details = get_app_details(app_id)
        app_name = app_details['name']
        if only_print_banners:
            banner_url = app_details['header_image']

            image_width = 150
            max_num_banners_per_row = 5  # only used with BBCode for now

            if use_markdown_syntax:
                # Markdown
                # Reference: https://stackoverflow.com/a/14747656
                image_link_str = '[<img alt="{}" src="{}" width="{}">]({})'
                print(image_link_str.format(app_name, banner_url, image_width, get_store_url(app_id)))
            else:
                # BBCode
                image_link_str = '[url={}][img="width:{}px;"]{}[/img][/url]'
                print(image_link_str.format(get_store_url(app_id), image_width, banner_url),
                      end='')
                # Line break every 5 lines
                if (rank + 1) % max_num_banners_per_row == 0:
                    print()

        else:
            # No banner, so that this is easier to read in Python console.
            print('{}) app: {} ({} @ {})'.format(rank + 1, app_id, app_name, get_store_url(app_id)))

        if rank >= (num_elements_displayed - 1):
            break
    return


def normalized(a, axis=-1, order=2):
    # Reference: https://stackoverflow.com/a/21032099
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def get_top_100_app_ids():
    data_request = dict()
    data_request['request'] = 'top100in2weeks'

    data = steamspypi.download(data_request)

    top_100_app_ids = list(data.keys())

    return top_100_app_ids


def load_benchmarked_app_ids(append_hard_coded_app_ids=True):
    # Reference: https://github.com/woctezuma/steam-descriptions/blob/master/benchmark_utils.py

    top_100_app_ids = get_top_100_app_ids()

    # Append hard-coded appIDs

    additional_app_ids = ['620', '364470', '504230', '583950', '646570', '863550', '794600', '814380']

    benchmarked_app_ids = top_100_app_ids
    if append_hard_coded_app_ids:
        for app_id in set(additional_app_ids).difference(top_100_app_ids):
            benchmarked_app_ids.append(app_id)

    return benchmarked_app_ids


def batch_retrieve_similar_features(query_app_ids=None,
                                    use_keras_features=True,
                                    use_cosine_similarity=True,
                                    print_banners=True,
                                    use_markdown_syntax=True,
                                    pooling=None,
                                    data_folder=None,
                                    images_are_store_banners=True):
    if query_app_ids is None:
        query_app_ids = load_benchmarked_app_ids()

    descriptor_database = None
    descriptor_img_id = None
    label_database = None

    feature_database_exists = True

    if use_keras_features:
        try:
            print('\n[pooling] {}'.format(pooling))
            label_database = np.load(get_label_database_filename(pooling))
        except FileNotFoundError or OSError:
            if pooling is None:
                feature_database_exists = False
            else:
                label_database = convert_label_database(pooling)

        keras_model, target_model_size = load_keras_model(include_top=False, pooling=pooling)

        if use_cosine_similarity:
            knn = NearestNeighbors(metric='cosine', algorithm='brute')
            knn.fit(label_database)
        else:
            knn = NearestNeighbors(algorithm='brute')
            knn.fit(label_database)
    else:
        try:
            descriptor_database = np.load(get_descriptor_database_filename())
            descriptor_img_id = np.load(get_descriptor_img_id_filename())
        except FileNotFoundError:
            feature_database_exists = False

        keras_model = None
        target_model_size = None
        knn = None

    if feature_database_exists:
        for query_app_id in query_app_ids:
            try:
                reference_app_id_counter = retrieve_similar_features(query_app_id,
                                                                     descriptor_database,
                                                                     descriptor_img_id,
                                                                     label_database,
                                                                     keras_model,
                                                                     target_model_size,
                                                                     pooling,
                                                                     knn,
                                                                     data_folder=data_folder,
                                                                     images_are_store_banners=images_are_store_banners)
            except FileNotFoundError:
                print('Query image not found: appID {} likely unavailable in this region.'.format(query_app_id))
                continue
            print_ranking(query_app_id, reference_app_id_counter, only_print_banners=print_banners,
                          use_markdown_syntax=use_markdown_syntax)

    return


if __name__ == '__main__':
    query_app_ids = None  # ['620', '364470', '504230', '583950', '646570', '863550', '794600']

    use_keras_features = True
    use_cosine_similarity = True
    print_banners = True
    use_markdown_syntax = True

    for pooling in [None]:  # , 'max', 'avg']:  # None or 'avg' or 'max'
        batch_retrieve_similar_features(query_app_ids,
                                        use_keras_features,
                                        use_cosine_similarity,
                                        print_banners,
                                        use_markdown_syntax,
                                        pooling,
                                        data_folder='128x128/',
                                        images_are_store_banners=True)
