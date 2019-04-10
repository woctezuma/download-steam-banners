# Download Steam Banners

[![Build status][build-image]][build]
[![Updates][dependency-image]][pyup]
[![Python 3][python3-image]][pyup]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]

This repository contains Python code to retrieve Steam games with similar store banners.

![Similar banners with cosine similarity and average pooling](https://github.com/woctezuma/download-steam-banners/wiki/img/LVUG4Gb.png)

## Requirements

-   Install the latest version of [Python 3.X](https://www.python.org/downloads/). For CNTK, you will need Python 3.6.
-   Install the required packages:

```bash
pip install -r requirements.txt
```

NB: For Windows, you may find a more recent version of OpenCV in [this repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/). Download it locally & install with:

```bash
pip install opencv_python-4.0.1-cp36-cp36m-win_amd64.whl
```

## Data

### Download Steam app details

The current code parses game names from app details. To provide access to app details:

1.   Download app details with [`steam-api`](https://github.com/woctezuma/steam-api) or directly from [`steam-api-data`](https://github.com/woctezuma/steam-api-data), 
2.   Fill in the path to `steam-api/data/appdetails/` in `download_steam_banners.py`.
If you use [PyCharm](https://www.jetbrains.com/pycharm/) on Windows, you could just mention your Windows username as follows:

```python
def get_user_name():
    user_name = 'Woctezuma' # <--- here
    return user_name
```

Alternatively, the code could be edited to rely on more accessible data, available via [SteamSpy API](https://github.com/woctezuma/steamspypi).

### Download Steam banners

There are 3 possibilities to download the Steam store banners of every Steam game:
1.   If you have already downloaded the app details with [`steam-api`](https://github.com/woctezuma/steam-api), then you could run `download_steam_banners.py`.
2.   Run [this IPython notebook](https://github.com/woctezuma/google-colab/blob/master/download_steam_banners.ipynb) which does not require to download app details: the URLs for Steam banners are similarly formatted and can be inferred from the appID alone.
3.   Download a data snapshot from February 2019, consisting of 31,723 Steam banners, with RGB channels, saved as .jpg:
     - [`download-steam-banners-data/`](https://github.com/woctezuma/download-steam-banners-data) for original images,
     - [`128x128.zip`](https://github.com/woctezuma/google-colab/tree/master/data) for images resized to 128x128 resolution.

Method 1 is the most exhaustive option: all the appIDs are listed via Steam API, and app details allow to check whether an appID matches a game, a DLC, a video, etc.

Method 2 is a nice trade-off: appIDs are listed via SteamSpy API, and they all match games. However, SteamSpy omits a few games based on their tags, etc.

Method 3 is the fastest option: you get a snapshot of downsampled banners.

## Usage

### Pre-processing

Store banners are resized to 128x128 with [`batch_resize_images.py`](batch_resize_images.py).

### Find games with similar Steam banners

Retrieve Steam games with similar store banners:
-   either based on [hashes](https://github.com/JohannesBuchner/imagehash) with `build_search_index.py` and `retrieve_similar_banners.py`,
-   or based on image features with `build_feature_index.py` and `retrieve_similar_features.py`:
    - either [ORB descriptors](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html),
    - or features extracted by [a neural net](https://keras.io/applications/#models-for-image-classification-with-weights-trained-on-imagenet).

## Results

In the results below, Steam banners are matched based on features from a neural net.
The features can be either concatenated, or merged via a pooling process (average or maximum pooling).
The similarity between features can be assessed with either cosine similarity or Minkowski distance.

An in-depth commentary is provided on the [Wiki](https://github.com/woctezuma/download-steam-banners/wiki/Commentary).
Overall, I would suggest to match features with:
-   cosine similarity, to avoid having to deal with weird matches of feature vectors with a norm close to zero,
-   either concatenation or average pooling: with concatenation, the banner layout greatly constrains the matching.

Caveat: the following pages contain a lot of images and might be slow to load depending on your Internet bandwith.

### Based on the concatenation of raw features

Results based on the concatenation of raw features from a neural net are shown with:
-   [cosine similarity](https://github.com/woctezuma/download-steam-banners/wiki/top_100_cosine_similarity),
-   [Minkowski distance](https://github.com/woctezuma/download-steam-banners/wiki/top_100_minkowski_distance).

### Based on the average pooling of features

Results based on the average pooling of features are shown with:
-   [cosine similarity](https://github.com/woctezuma/download-steam-banners/wiki/top_100_cosine_similarity_with_average_pooling),
-   [Minkowski distance](https://github.com/woctezuma/download-steam-banners/wiki/top_100_minkowski_distance_with_average_pooling).

### Based on the maximum pooling of features

Results based on the maximum pooling of features are shown with:
-   [cosine similarity](https://github.com/woctezuma/download-steam-banners/wiki/top_100_cosine_similarity_with_max_pooling),
-   [Minkowski distance](https://github.com/woctezuma/download-steam-banners/wiki/top_100_minkowski_distance_with_max_pooling).

## References

-   [`download-steam-screenshots`](https://github.com/woctezuma/download-steam-screenshots): retrieve Steam games with similar store **screenshots**,
-   [Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.](https://github.com/richzhang/PerceptualSimilarity)
-   [Image hashes](https://github.com/JohannesBuchner/imagehash)
-   [Another Github repository about image similarity](https://github.com/ankonzoid/artificio)

<!-- Definitions -->

[build]: <https://travis-ci.org/woctezuma/download-steam-banners>
[build-image]: <https://travis-ci.org/woctezuma/download-steam-banners.svg?branch=master>

[pyup]: <https://pyup.io/repos/github/woctezuma/download-steam-banners/>
[dependency-image]: <https://pyup.io/repos/github/woctezuma/download-steam-banners/shield.svg>
[python3-image]: <https://pyup.io/repos/github/woctezuma/download-steam-banners/python-3-shield.svg>

[codecov]: <https://codecov.io/gh/woctezuma/download-steam-banners>
[codecov-image]: <https://codecov.io/gh/woctezuma/download-steam-banners/branch/master/graph/badge.svg>

[codacy]: <https://www.codacy.com/app/woctezuma/download-steam-banners>
[codacy-image]: <https://api.codacy.com/project/badge/Grade/c3ff7d48630544209f3adf29b03e1048>
