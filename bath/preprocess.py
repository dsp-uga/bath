"""
The preprocess module contains several helper functions for loading and preprocessing sets of images
"""

from skimage import io
import numpy as np
from glob import glob
import os
import json

from bath.Dataset import Dataset


def load(name, location, greyscale=True):
    """
    Loads a single example from disk and converts it to a bath.Dataset object.
    This function assumes a directory structure similar to that of the codeneuro datasets.  That is, the input directory
    (location) should contain an "images" subdirectory and optionally a "regions" subdirectory.
    If a regions subdirectory cannot be found in the given location, then regions will not be loaded.

    :param name: the logical name of the dataset
    :param location: the directory where the data are located - should have a subdirectory "images"
    :param greyscale: load the images as greyscale.  Defaults to true
    :return: bath.Dataset representing this sequence of images
    """
    # make sure the input directory exists:
    assert(os.path.exists(os.path.join(location, "images")))

    # load images into a np array
    file_pattern = os.path.join(location, "images", "*.tiff")
    files = sorted(glob(file_pattern))
    images = np.array([io.imread(f, as_grey=greyscale) for f in files])

    dataset = Dataset(name, images)

    # load regions if available
    has_regions = os.path.exists(os.path.join(location, 'regions'))
    if has_regions:
        regions_file = os.path.join(location, 'regions', 'regions.json')
        with open(regions_file) as file:
            regions = json.load(file)
            # contents = file.read()

        # regions = json.loads(contents)
        dataset.set_regions(regions)

    return dataset
