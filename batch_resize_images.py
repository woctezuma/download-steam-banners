import glob
from pathlib import Path
from time import time

import cv2
import numpy as np


def center_crop(img, new_width=None, new_height=None):
    # Reference: https://stackoverflow.com/a/32385865/

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img


def resize_all_images(
    output_shape=None,
    display_images=False,
    verbose_interval=1000,
    sub_folder=None,
    apply_center_crop=False,
    apply_resize=True,
):
    if output_shape is None:
        output_shape = [128, 128]

    output_folder = 'x'.join([str(i) for i in output_shape]) + '/'
    Path(output_folder).mkdir(exist_ok=True)

    input_folder = 'data/'
    if sub_folder is not None:
        input_folder += sub_folder
    if not input_folder.endswith('/'):
        input_folder += '/'

    file_paths = glob.glob(input_folder + '*.jpg')

    start = time()

    for counter, input_path in enumerate(file_paths):
        file_name = Path(input_path).name
        output_path = output_folder + file_name

        if not Path(output_path).exists():
            image = cv2.imread(input_path)

            center_cropped_image = center_crop(image) if apply_center_crop else image

            if apply_resize:
                resized_image = cv2.resize(center_cropped_image, tuple(output_shape))
            else:
                resized_image = center_cropped_image

            cv2.imwrite(output_path, resized_image)

            if (counter + 1) % verbose_interval == 0:
                print(
                    '[{}/{}] Elapsed time: {:.2f} seconds'.format(
                        counter + 1,
                        len(file_paths),
                        time() - start,
                    ),
                )

                if display_images:
                    cv2.imshow('image', image)
                    cv2.imshow('resized_image', resized_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                start = time()

    return


if __name__ == '__main__':
    resize_all_images()
