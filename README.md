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

First, download the app details with [`steam-api`](https://github.com/woctezuma/steam-api).

Then, run `download_steam_banners.py` to download the store banners of every Steam game.

Finally, retrieve Steam games with similar store banners:
-   either based on [hashes](https://github.com/JohannesBuchner/imagehash) with `build_search_index.py` and `retrieve_similar_banners.py`,
-   or based on image features ([ORB descriptors](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html) or features from [a neural net](https://keras.io/applications/#models-for-image-classification-with-weights-trained-on-imagenet)) with `build_feature_index.py` and `retrieve_similar_features.py`.

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
