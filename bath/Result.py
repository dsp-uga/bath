"""
Object-oriented interface for the results output by a model
Includes convenience methods for computing accuracy combining many results, and writing results to a JSON file
"""

import json
import neurofinder


class Result(object):
    def __init__(self, name, regions):
        """
        Result objects represent the result of running a model against a single example
        :param name: name of the dataset corresponding to these results
        :param regions: list of dictionaries specifying the ROI
                        Each dictionary should contain a "coordinates" key that gives a list of (x, y) coordinates
        """
        self.name = name
        self.raw_regions = regions
        regions_json = json.dumps(regions)
        self.regions = neurofinder.load(regions_json)

    def f_score(self, ground_truth):
        """
        Computes the f_score
        F = 2 * (recall * precision) / (recall + precision)
        given ground truth regions

        :param ground_truth: regions object created by the regional package (or by neurofinder.load)
        :return: F-score obtained by comparing the ground truth to the regions in this Result
        """
        recall, precision = neurofinder.centers(ground_truth, self.regions)
        combined = 2 * (recall * precision) / (recall + precision)
        return combined

    def to_json(self):
        """
        Converts the results to a json string
        [{"coordinates": [[x, y], [x, y], ...]}, {"coordinates": [[x, y], [x, y], ...]}, ...]

        :return:
        """
        return json.dumps(self.raw_regions)

    def to_output(self):
        """
        Converts the Dataset object to a dictionary of the form
        {
            "dataset": <name>,
            "regions": [{"coordinates": [[x, y], [x, y], ...]}, {"coordinates": [[x, y], [x, y], ...]}, ...]
        }

        :return: dict with keys "dataset" and "regions"
        """
        return {
            "dataset": self.name,
            "regions": self.to_json()
        }
