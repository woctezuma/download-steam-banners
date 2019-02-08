from time import time

import cv2 as cv
import numpy as np
from keras.applications.nasnet import NASNetMobile
from keras.applications.nasnet import decode_predictions
from keras.applications.nasnet import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt

from build_search_index import list_app_ids, app_id_to_image_filename
from download_steam_banners import get_app_details


def get_descriptor_database_filename():
    descriptor_database_filename = 'descriptor_database.npy'
    return descriptor_database_filename


def get_descriptor_img_id_filename():
    descriptor_img_id_filename = 'descriptor_img_id.npy'
    return descriptor_img_id_filename


def load_keras_model(include_top=True):
    model = NASNetMobile(include_top=include_top)
    target_model_size = (224, 224)

    return model, target_model_size


def label_image(image, model, verbose=False):
    # Reference: https://github.com/glouppe/blackbelt/

    # convert the image pixels to a numpy array
    image = img_to_array(image)

    # reshape data for the model
    image = np.expand_dims(image, axis=0)

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    if verbose and model.output_names[0] == 'predictions':
        # convert the probabilities to class labels
        labels = decode_predictions(yhat)

        # retrieve the most likely result, e.g. highest probability
        label = labels[0][0]

        # print the classification
        print('%s (%.2f%%)' % (label[1], label[2] * 100))

    return yhat


def build_feature_index(verbose=False):
    # Reference: https://docs.opencv.org/4.0.1/dc/dc3/tutorial_py_matcher.html

    app_ids = list_app_ids()

    num_games = len(app_ids)

    start = time()

    # Initiate ORB detector
    orb = cv.ORB_create()

    # Load the model
    model, target_model_size = load_keras_model()

    descriptor_database = None
    descriptor_img_id = None

    for (counter, app_id) in enumerate(app_ids):

        if (counter % 1000) == 0:
            print('[{}/{}] appID = {}'.format(counter, num_games, app_id))
            print('Elapsed time: {:.2f} s'.format(time() - start))
            start = time()

        image_filename = app_id_to_image_filename(app_id)
        img = cv.imread(image_filename, cv.IMREAD_COLOR)

        image = load_img(image_filename, target_size=target_model_size)
        label_image(image, model)

        # find the keypoints and descriptors with ORB
        kp, des = orb.detectAndCompute(img, None)

        current_des = np.array(des)
        try:
            current_img_id = np.zeros((des.shape[0], 1)) + counter
        except AttributeError:
            continue

        if descriptor_database is None:
            descriptor_database = current_des.copy()
            descriptor_img_id = current_img_id.copy()
        else:
            descriptor_database = np.vstack((descriptor_database, current_des))
            descriptor_img_id = np.vstack((descriptor_img_id, current_img_id))

        if verbose:
            app_details = get_app_details(app_id)
            app_name = app_details['name']
            print('AppID = {} ({})'.format(app_id, app_name))

            # draw only keypoints location,not size and orientation
            # If the OpenCV build for Python is fixed:
            # Reference: https://github.com/skvark/opencv-python/issues/168
            # img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
            # Otherwise:
            img2 = img.copy()
            for marker in kp:
                img2 = cv.drawMarker(img2,
                                     tuple(int(i) for i in marker.pt), color=(0, 255, 0), markerType=cv.MARKER_DIAMOND)
            plt.imshow(img2)
            plt.show()

    np.save(get_descriptor_database_filename(), descriptor_database)
    np.save(get_descriptor_img_id_filename(), descriptor_img_id)

    return


if __name__ == '__main__':
    build_feature_index(verbose=False)
