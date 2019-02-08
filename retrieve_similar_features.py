from collections import Counter
from time import time

import cv2 as cv
import numpy as np
from keras.preprocessing.image import load_img

from build_feature_index import get_descriptor_database_filename, get_descriptor_img_id_filename
from build_feature_index import get_label_database_filename, load_keras_model, label_image
from build_search_index import app_id_to_image_filename, list_app_ids
from download_steam_banners import get_app_details
from retrieve_similar_banners import get_store_url


def retrieve_similar_features(query_app_id, descriptor_database=None, descriptor_img_id=None,
                              label_database=None, keras_model=None, target_model_size=None):
    image_filename = app_id_to_image_filename(query_app_id)

    if keras_model is not None:
        image = load_img(image_filename, target_size=target_model_size)
        query_des = label_image(image, keras_model)  # runtime: 1 second

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

    if descriptor_database is None:
        descriptor_database = np.load(get_descriptor_database_filename())

    if descriptor_img_id is None:
        descriptor_img_id = np.load(get_descriptor_img_id_filename())

    if label_database is None:
        label_database = np.load(get_label_database_filename())

    app_ids = list_app_ids()

    if keras_model is not None:
        row_no = app_ids.index(query_app_id)
        trimmed_descriptor_database = label_database.copy()
        trimmed_descriptor_database[row_no, :] = 0

        # For FLANN, the query and the database should have the same dtype 'float32'.
        trimmed_descriptor_database = trimmed_descriptor_database.astype('float32')

        num_neighbors = 10
    else:
        # Remove the features of the query from the database
        ind, _ = np.where(descriptor_img_id == app_ids.index(query_app_id))
        trimmed_descriptor_database = descriptor_database.copy()
        trimmed_descriptor_database[ind] = 0

        num_neighbors = 2

    start = time()
    matches = flann.knnMatch(query_des, trimmed_descriptor_database, k=num_neighbors)
    print('Elapsed time: {:.2f} s'.format(time() - start))

    if keras_model is not None:
        # When we use the Keras model, a Steam banner is represented by only ONE feature, hence the use of 'matches[0]'.
        reference_app_id_counter = [app_ids[element.trainIdx] for element in matches[0]]
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


def print_ranking(query_app_id, reference_app_id_counter, num_elements_displayed=10):
    app_details = get_app_details(query_app_id)
    app_name = app_details['name']
    print('\nQuery appID: {} ({} @ {})'.format(query_app_id, app_name, get_store_url(query_app_id)))

    for rank, app_id in enumerate(reference_app_id_counter):
        app_details = get_app_details(app_id)
        app_name = app_details['name']
        print('{}) app: {} ({} @ {})'.format(rank + 1, app_id, app_name, get_store_url(app_id)))

        if rank >= (num_elements_displayed - 1):
            break
    return


if __name__ == '__main__':
    descriptor_database = np.load(get_descriptor_database_filename())
    descriptor_img_id = np.load(get_descriptor_img_id_filename())
    label_database = np.load(get_label_database_filename())

    query_app_ids = ['620', '364470', '504230', '583950', '646570', '863550', '794600']

    use_keras_features = True

    if use_keras_features:
        keras_model, target_model_size = load_keras_model(include_top=False)
    else:
        keras_model = None
        target_model_size = None

    for query_app_id in query_app_ids:
        reference_app_id_counter = retrieve_similar_features(query_app_id, descriptor_database, descriptor_img_id,
                                                             label_database, keras_model, target_model_size)
        print_ranking(query_app_id, reference_app_id_counter)
