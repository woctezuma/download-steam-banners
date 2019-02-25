# Download Steam Banners

[![Build status][build-image]][build]
[![Updates][dependency-image]][pyup]
[![Python 3][python3-image]][pyup]
[![Code coverage][codecov-image]][codecov]
[![Code Quality][codacy-image]][codacy]

This repository contains Python code to retrieve Steam games with similar store banners.

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

## Usage

### Download Steam banners

There are 3 possibilities to download the Steam store banners of every Steam game:
1.   If you have already downloaded the app details with [`steam-api`](https://github.com/woctezuma/steam-api), then you could run `download_steam_banners.py`.
2.   Run [this IPython notebook](https://github.com/woctezuma/google-colab/blob/master/download_steam_banners.ipynb) which does not require to download app details: the URLs for Steam banners are similarly formatted and can be inferred from the appID alone.
3.   Download a data snapshot [`128x128.zip`](https://github.com/woctezuma/google-colab/tree/master/data) from early February 2019, consisting of 31,723 Steam banners, resized to 128x128 resolution and saved as .jpg files with RGB channels.

Method 1 is the most exhaustive option: all the appIDs are listed via Steam API, and app details allow to check whether an appID matches a game, a DLC, a video, etc.

Method 2 is a nice trade-off: appIDs are listed via SteamSpy API, and they all match games. However, SteamSpy omits a few games based on their tags, etc.

Method 3 is the fastest option: you get a snapshot of downsampled banners.

### Find games with similar Steam banners

Finally, retrieve Steam games with similar store banners:
-   either based on [hashes](https://github.com/JohannesBuchner/imagehash) with `build_search_index.py` and `retrieve_similar_banners.py`,
-   or based on image features ([ORB descriptors](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html) or features from [a neural net](https://keras.io/applications/#models-for-image-classification-with-weights-trained-on-imagenet)) with `build_feature_index.py` and `retrieve_similar_features.py`.

## References

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
