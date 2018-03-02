"""
The postprocess module contains utility functions to write results to a file
"""

import os
import json


def write_results(results, output_dir, name="output.json"):
    """
    Writes the given set of bath.Result objects to the given path

    :param results: list of bath.Result objects
    :param output_dir: directory where output will be written
    :param name: name of the output file
    :return: None
    """
    # ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    outpath = os.path.join(output_dir, name)

    output = [result.to_output() for result in results]
    with open(outpath, 'w') as outfile:
        json.dump(output, outfile)
