import glob
from pathlib import Path
from time import time

import cv2


def resize_all_images(output_shape=None, display_images=False, verbose_interval=1000):
    if output_shape is None:
        output_shape = [128, 128]

    output_folder = 'x'.join([str(i) for i in output_shape]) + '/'
    Path(output_folder).mkdir(exist_ok=True)

    file_paths = glob.glob('data/*.jpg')

    start = time()

    for counter, input_path in enumerate(file_paths):
        file_name = Path(input_path).name
        output_path = output_folder + file_name

        if not Path(output_path).exists():

            image = cv2.imread(input_path)
            resized_image = cv2.resize(image, tuple(output_shape))
            cv2.imwrite(output_path, resized_image)

            if (counter + 1) % verbose_interval == 0:
                print('[{}/{}] Elapsed time: {:.2f} seconds'.format(counter + 1, len(file_paths), time() - start))

                if display_images:
                    cv2.imshow('image', image)
                    cv2.imshow('resized_image', resized_image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                start = time()

    return


if __name__ == '__main__':
    resize_all_images()
