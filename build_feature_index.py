import pathlib
from time import time

import cv2 as cv
import numpy as np
from keras.applications.mobilenet import MobileNet, decode_predictions, preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt

from build_search_index import app_id_to_image_filename, list_app_ids
from download_steam_banners import get_app_details


def get_features_folder_name():
    features_folder_name = 'features/'
    # Reference of the following line: https://stackoverflow.com/a/14364249
    pathlib.Path(features_folder_name).mkdir(exist_ok=True)
    return features_folder_name


def get_descriptor_database_filename():
    descriptor_database_filename = (
        get_features_folder_name() + 'descriptor_database.npy'
    )
    return descriptor_database_filename


def get_descriptor_img_id_filename():
    descriptor_img_id_filename = get_features_folder_name() + 'descriptor_img_id.npy'
    return descriptor_img_id_filename


def get_label_database_filename(pooling=None):
    pooling_str = '' if pooling is None else '.' + pooling

    label_database_filename = (
        get_features_folder_name() + 'label_database' + pooling_str + '.npy'
    )
    return label_database_filename


def get_frozen_app_ids_filename():
    frozen_app_ids_filename = get_features_folder_name() + 'frozen_app_ids.txt'
    return frozen_app_ids_filename


def get_frozen_app_ids():
    with open(get_frozen_app_ids_filename()) as f:
        frozen_app_ids = [app_id.strip() for app_id in f.readlines()]

    return frozen_app_ids


def freeze_app_ids(app_ids):
    with open(get_frozen_app_ids_filename(), 'w') as f:
        for app_id in app_ids:
            f.write(str(app_id) + '\n')

    return


def convert_label_database(target_pooling=None):
    # Convert from a database of features obtained with:
    # - 'include_top' set to False
    # - and 'pooling' set to None
    # to the database of features which would have been obtained with:
    # - 'include_top' set to False
    # - and 'pooling' set to 'max' or to 'avg'

    X = np.load(get_label_database_filename(pooling=None))

    # Feature shape before flattening
    original_feature_shape = [
        4,
        4,
        256,
    ]  # Caveat: hard-coded for MobileNet with alpha=0.25 and input_shape=(128,128,3)

    if target_pooling == 'max':
        X_target = np.zeros((X.shape[0], original_feature_shape[-1]))
        for row, x in enumerate(X):
            X_target[row, :] = np.max(x.reshape(original_feature_shape), axis=(0, 1))
    elif target_pooling == 'avg':
        X_target = np.zeros((X.shape[0], original_feature_shape[-1]))
        for row, x in enumerate(X):
            X_target[row, :] = np.mean(x.reshape(original_feature_shape), axis=(0, 1))
    else:
        X_target = X

    return X_target


def load_keras_model(include_top=True, pooling='avg'):
    # The function argument allows to choose whether to include the last model layer for label prediction.

    # Reference: https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py
    alpha_value = 0.25
    target_model_size = (128, 128)

    num_channels = 3
    # Image data format: channels last
    input_shape = (*list(target_model_size), num_channels)

    if include_top:
        model = MobileNet(
            include_top=include_top,
            alpha=alpha_value,
            input_shape=input_shape,
        )
    else:
        model = MobileNet(
            include_top=include_top,
            pooling=pooling,
            alpha=alpha_value,
            input_shape=input_shape,
        )

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
        print('{} ({:.2f}%)'.format(label[1], label[2] * 100))

    return yhat


def build_feature_index(
    verbose=False,
    save_keras_output=False,
    include_top=True,
    pooling=None,
    data_folder=None,
):
    # Reference: https://docs.opencv.org/4.0.1/dc/dc3/tutorial_py_matcher.html

    app_ids = list_app_ids(data_folder=data_folder)

    num_games = len(app_ids)

    start = time()

    # Initiate ORB detector
    orb = cv.ORB_create()

    # Load the model
    model, target_model_size = load_keras_model(include_top, pooling=pooling)

    descriptor_database = None
    descriptor_img_id = None
    Y_hat = None
    frozen_Y_hat = None
    frozen_app_ids = None

    if save_keras_output:
        Y_hat = np.zeros((num_games, np.product(model.output_shape[1:])))

        try:
            frozen_Y_hat = np.load(get_label_database_filename(pooling))
        except FileNotFoundError:
            frozen_Y_hat = None

        try:
            frozen_app_ids = get_frozen_app_ids()
        except FileNotFoundError:
            frozen_app_ids = None

    for counter, app_id in enumerate(app_ids):
        if frozen_Y_hat is not None and frozen_app_ids is not None:
            # Avoid re-computing values of Y_hat which were previously computed and saved to disk, then recently loaded
            try:
                frozen_counter = frozen_app_ids.index(app_id)

                if any(frozen_Y_hat[frozen_counter, :] != 0):
                    Y_hat[counter, :] = frozen_Y_hat[frozen_counter, :]
                    continue

            except ValueError:
                pass

        if (counter % 1200) == 0:
            print(f'[{counter}/{num_games}] appID = {app_id}')
            print(
                'Elapsed time for computing image features: {:.2f} s'.format(
                    time() - start,
                ),
            )
            start = time()

            if Y_hat is not None:
                np.save(get_label_database_filename(pooling), Y_hat)
                freeze_app_ids(app_ids)
                print(
                    'Elapsed time for saving the result to disk: {:.2f} s'.format(
                        time() - start,
                    ),
                )
                start = time()

        image_filename = app_id_to_image_filename(app_id, data_folder=data_folder)

        if Y_hat is not None:
            image = load_img(image_filename, target_size=target_model_size)
            yhat = label_image(image, model)  # runtime: 1 second
            Y_hat[counter, :] = yhat.flatten()
        else:
            img = cv.imread(image_filename, cv.IMREAD_COLOR)

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
                print(f'AppID = {app_id} ({app_name})')

                # draw only keypoints location,not size and orientation
                # If the OpenCV build for Python is fixed:
                # Reference: https://github.com/skvark/opencv-python/issues/168
                # img2 = cv.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
                # Otherwise:
                img2 = img.copy()
                for marker in kp:
                    img2 = cv.drawMarker(
                        img2,
                        tuple(int(i) for i in marker.pt),
                        color=(0, 255, 0),
                        markerType=cv.MARKER_DIAMOND,
                    )
                plt.imshow(img2)
                plt.show()

    if Y_hat is not None:
        np.save(get_label_database_filename(pooling), Y_hat)
        freeze_app_ids(app_ids)
    else:
        np.save(get_descriptor_database_filename(), descriptor_database)
        np.save(get_descriptor_img_id_filename(), descriptor_img_id)

    return


if __name__ == '__main__':
    pooling = None  # None or 'avg' or 'max'
    build_feature_index(
        verbose=False,
        save_keras_output=True,
        include_top=False,
        pooling=pooling,
        data_folder='128x128/',
    )
