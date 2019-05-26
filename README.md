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

On Windows, the most recent version of OpenCV can also be downloaded from [this repository](https://www.lfd.uci.edu/~gohlke/pythonlibs/). To install it:

```bash
pip install opencv_python-4.0.1-cp36-cp36m-win_amd64.whl
```

## Data

A data snapshot from February 2019 is available in:
-   [`download-steam-banners-data/`](https://github.com/woctezuma/download-steam-banners-data) for the 31,724 original RBG images,
-   [`128x128.zip`](https://github.com/woctezuma/google-colab/tree/master/data) for images resized to 128x128 resolution.

Otherwise, you would have to:
1.   download app details with [`steam-api`](https://github.com/woctezuma/steam-api),
2.   parse app details to find the banner URL of each game,
3.   download the banners with [`download_steam_banners.py`](download_steam_banners.py).

Alternatively, run [this IPython notebook](https://github.com/woctezuma/google-colab/blob/master/download_steam_banners.ipynb). 
A list of of appIDs (tied to games) is first downloaded via [SteamSpy API](https://github.com/woctezuma/steamspypi).
Then banner URLs are directly inferred from appIDs, without relying on app details.

## Usage

Store banners are resized to 128x128 with [`batch_resize_images.py`](batch_resize_images.py).

To retrieve Steam games with similar store banners, image features are:
1.   extracted by [a neural net](https://keras.io/applications/#models-for-image-classification-with-weights-trained-on-imagenet) with [`build_feature_index.py`](build_feature_index.py),
2.   either concatenated, or merged via a pooling process (average or maximum pooling),
3.   compared based on cosine similarity or Minkowski distance with [`retrieve_similar_features.py`](retrieve_similar_features.py).

Alternatively:
-   [image hashes](https://github.com/JohannesBuchner/imagehash) could be computed with `build_search_index.py` and `retrieve_similar_banners.py`,
-   different image features could be computed, e.g. [ORB descriptors](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html) with `build_feature_index.py` and `retrieve_similar_features.py`.

## Caveat

Fill-in the path to `steam-api/data/appdetails/` in `download_steam_banners.py`.
If you use [PyCharm](https://www.jetbrains.com/pycharm/) on Windows, you could just mention your Windows username as follows:

```python
def get_user_name():
    user_name = 'Woctezuma' # <--- here
    return user_name
```

## Results

### Similar games

Results obtained with a neural net ([MobileNet](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py)) are shown [on the Wiki](https://github.com/woctezuma/download-steam-banners/wiki).

Results obtained with alternative methods are not shown:
-   image hashes are mostly useful to detect duplicates,
-   ORB descriptors seem less relevant than the features obtained with MobileNet. 

An in-depth commentary is provided [on the Wiki](https://github.com/woctezuma/download-steam-banners/wiki/Commentary).
Overall, I would suggest to match features with:
-   cosine similarity, to avoid having to deal with weird matches of feature vectors with a norm close to zero,
-   either [concatenation](https://github.com/woctezuma/download-steam-banners/wiki/top_100_cosine_similarity) or [average pooling](https://github.com/woctezuma/download-steam-banners/wiki/top_100_cosine_similarity_with_average_pooling): with concatenation, the banner layout greatly constrains the matching.

### Unique games

It is possible to highlight games with *unique* store banners, by applying a threshold to similarity values output by the algorithm.
This is done in [`find_unique_games.py`](find_unique_games.py):
-   cosine similarity is used to compare features,
-   a game is *unique* if the similarity score between a query game and its most similar game (other than itself) is lower than or equal to an arbitrary threshold.

Results are shown [here](https://github.com/woctezuma/download-steam-banners/wiki/Unique_Games).

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
