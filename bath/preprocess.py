"""
The preprocess module contains several helper functions for loading and preprocessing sets of images
"""

from skimage import io
import numpy as np
from glob import glob
import os
import json

from bath.Dataset import Dataset


def name_to_directory(name):
    return f'neurofinder.{name}'


def load(name, base_dir, greyscale=True):
    """
    Loads a single example from disk and converts it to a bath.Dataset object.
    This function assumes a directory structure similar to that of the codeneuro datasets.  That is, the input directory
    (base_dir) should contain an "neurofinder.<name>" subdirectory which contains an "images" subdirectory and an
    optional "regions" subdirectory.
    If a regions subdirectory cannot be found in the given location, then regions will not be loaded.

    Example: load('00.00', './data') will load the dataset contained in the directory './data/neurofinder.00.00/images'
    and, if found, the regions contained in the directory './data/neurofinder.00.00/regions'

    :param name: the logical name of the dataset (will be converted to the directory name "neurofinder.<name>")
    :param base_dir: the directory where the the dataset folder is located
    :param greyscale: load the images as greyscale.  Defaults to true
    :return: bath.Dataset representing this sequence of images
    """
    data_directory = os.path.join(base_dir, name_to_directory(name))
    # make sure the input directory exists:
    assert(os.path.exists(os.path.join(data_directory, "images")))

    # load images into a np array
    file_pattern = os.path.join(data_directory, "images", "*.tiff")
    files = sorted(glob(file_pattern))
    images = np.array([io.imread(f, as_grey=greyscale) for f in files])

    dataset = Dataset(name, images)

    # load regions if available
    has_regions = os.path.exists(os.path.join(data_directory, 'regions'))
    if has_regions:
        regions_file = os.path.join(data_directory, 'regions', 'regions.json')
        with open(regions_file) as file:
            regions = json.load(file)

        dataset.set_regions(regions)

    return dataset
