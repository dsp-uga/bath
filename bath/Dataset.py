"""
Object-oriented interface for collections of images
"""

import numpy as np


class Dataset(object):

    def __init__(self, name, images, regions=None):
        """
        Dataset objects represent a sequence of images which make up a single example
        Each image is represented as a numpy ndarray
        The "shape" of the dataset is the shape of the images in the dataset, represented by [rows, cols, [channels]]

        :param name: the name of the dataset (usually 00.00, 01.01, etc)
        :param images: an array of images represented as numpy ndarray
        :param regions: optional array of dictionaries, each representing a segmented region-of-interest
        """
        self.name = name
        self.images = np.array(images)
        self.regions = np.array(regions)
        self.shape = None
        if len(self.images):
            self.shape = images[0].shape

    def get_data(self):
        """
        Syntactic sugar for accessing self.images
        :return: array of images in this dataset
        """
        return self.images

    def set_regions(self, regions):
        """
        Set the list of regions
        :param regions: array of dictionaries, each representing a segmented region-of-interest
        :return: None
        """
        self.regions = regions

    def to_output(self):
        """
        Converts the Dataset object to a dictionary of the form
        {
            "dataset": <name>,
            "regions": [{"coordinates": [[x, y], [x, y], ...]}, {"coordinates": [[x, y], [x, y], ...]}, ...]
        }

        :return: dict with keys "dataset" and "regions"
        """
        # TODO
        pass
