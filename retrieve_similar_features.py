from collections import Counter

import cv2 as cv
import numpy as np

from build_feature_index import get_descriptor_database_filename, get_descriptor_img_id_filename
from build_search_index import app_id_to_image_filename


def retrieve_similar_features(query_app_id, min_match_count=10):
    image_filename = app_id_to_image_filename(query_app_id)
    query_img = cv.imread(image_filename, cv.IMREAD_COLOR)

    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with ORB
    query_kp, query_des = orb.detectAndCompute(query_img, None)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)

    descriptor_database = np.load(get_descriptor_database_filename())
    descriptor_img_id = np.load(get_descriptor_img_id_filename())

    matches = flann.knnMatch(query_des, descriptor_database, k=2)

    # store all the good matches as per Lowe's ratio test.
    good_matches = []
    for t in matches:
        try:
            if t[0].distance < 0.7 * t[1].distance:
                good_matches.append(t[0])
        except IndexError:
            pass

    good_img_ids = [good_match.trainIdx for good_match in good_matches]

    reference_img_id_counter = Counter(good_img_ids)

    return reference_img_id_counter


if __name__ == '__main__':
    reference_img_id_counter = retrieve_similar_features(query_app_id='863550')
