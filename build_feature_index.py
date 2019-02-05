from time import time

import cv2 as cv
from matplotlib import pyplot as plt

from build_search_index import list_app_ids, app_id_to_image_filename
from download_steam_banners import get_app_details


def get_feature_index_filename():
    feature_index_filename = 'feature_index.json'
    return feature_index_filename


def build_feature_index(verbose=False):
    app_ids = list_app_ids()

    num_games = len(app_ids)

    start = time()

    # Initiate ORB detector
    orb = cv.ORB_create()

    for (counter, app_id) in enumerate(sorted(app_ids, key=int)):

        if (counter % 1000) == 0:
            print('[{}/{}] appID = {}'.format(counter, num_games, app_id))
            print('Elapsed time: {:.2f} s'.format(time() - start))
            start = time()

        image_filename = app_id_to_image_filename(app_id)
        img = cv.imread(image_filename, cv.IMREAD_COLOR)

        # find the keypoints and descriptors with ORB
        kp, des = orb.detectAndCompute(img, None)

        if verbose:
            app_details = get_app_details(app_id)
            app_name = app_details['name']
            print('AppID = {} ({})'.format(app_id, app_name))

            # draw only keypoints location,not size and orientation
            img2 = img
            for marker in kp:
                img2 = cv.drawMarker(img2,
                                     tuple(int(i) for i in marker.pt), color=(0, 255, 0), markerType=cv.MARKER_DIAMOND)
            plt.imshow(img2), plt.show()

    return feature_index


if __name__ == '__main__':
    feature_index = build_feature_index(verbose=True)
