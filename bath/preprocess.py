"""
The preprocess module contains several helper functions for loading and preprocessing sets of images

Credit to
https://github.com/codeneuro/neurofinder-python and https://github.com/freeman-lab/regional
for loading of regions from json files
"""

from skimage import io
import numpy as np
from glob import glob
import os
from neurofinder import load as load_regions

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

    # load regions if available
    regions = None
    has_regions = os.path.exists(os.path.join(data_directory, 'regions'))
    if has_regions:
        regions_file = os.path.join(data_directory, 'regions', 'regions.json')
        regions = load_regions(regions_file)

    dataset = Dataset(name, images, regions)

    return dataset


def compute_summary(images, type='mean'):
    """
    Compute a summary image
    Type should be one of 'mean', 'max', or 'min'

    :param images: t x w x h ndarray of images
    :param type: the type of summary to compute ['mean', 'max', 'min']
    :return: a single w x h array representing the resulting image
    """
    if type == 'mean':
        result = images.sum(axis=0) / images.shape[0]
    elif type == 'max':
        result = np.amax(images, axis=0)
    elif type == 'min':
        result = np.amin(images, axis=0)
    else:
        result = images[0]

    return result


def normalize(image):
    """
    Normalizes an image by subtracting the mean and dividing by the standard deviation

    :param image: w x h ndarray representing the image to normalize
    :return: w x h ndarray representing the normalized image
    """
    mean = image.sum() / (image.shape[0] + image.shape[1])
    sigma = np.std(image)
    normalized = (image - mean) / sigma

    return normalized
