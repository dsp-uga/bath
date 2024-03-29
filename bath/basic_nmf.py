"""
Neuron segmentation using Non-negative Matrix Factorization (NMF)

We use the NMF model from the thunder-extraction package:
https://github.com/thunder-project/thunder-extraction

The codeneuro localnmf example was also helpful:
https://gist.github.com/freeman-lab/330183fdb0ea7f4103deddc9fae18113
"""

from extraction import NMF
from bath import preprocess, postprocess
from bath.Result import Result
import numpy as np


def main(datasets, base_dir, output_dir="output/nmf", gaussian_blur=0, verbose=False,
         n_components=5, max_iter=20, threshold=99, overlap=0.1, chunk_size=(32, 32), padding=(20, 20), merge_iter=5):
    """
    Performs neuron segementation using the NMF implementation provided by thunder-extraction
    Results will be written to <output_dir>/00.00-output.json

    :param datasets: list of datasets (by name) to generate results for
    :param base_dir: directory that contains the datasets
    :param output_dir: directory where output file should be written
    :param k: number of components to estimate per block
    :param threshold: value for thresholding (higher means more thresholding)
    :param overlap: value for determining whether to merge (higher means fewer merges)
    :param chunk_size: process images in chunks of this size
    :param padding: add this much padding to each chunk
    :param merge_iter: number of iterations to perform when merging regions
    :return: array of bath.Result objects representing the result on each dataset
    """
    results = []
    for dataset_name in datasets:
        if verbose: print("Processing dataset %s" % dataset_name)
        dataset = preprocess.load(dataset_name, base_dir)
        if verbose: print("Dataset loaded.")

        if gaussian_blur > 0:
            chunks = np.array_split(dataset.images, 30)
            summaries = np.array(list(map(preprocess.compute_summary, chunks)))
            # summaries =
            dataset.images = preprocess.gaussian_blur(summaries, gaussian_blur)

        model = NMF(k=n_components, max_iter=max_iter, percentile=threshold, overlap=overlap, min_size=20)
        model = model.fit(dataset.images, chunk_size=chunk_size, padding=padding)
        merged = model.merge(overlap=overlap, max_iter=merge_iter, k_nearest=20)
        regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]

        result = Result(name=dataset_name, regions=regions)
        results.append(result)

        if verbose: print("Done with dataset %s" % dataset_name)

        if dataset.has_ground_truth() and verbose:
            f_score = result.f_score(dataset.true_regions)
            print("Combined score for dataset %s: %0.4f" % (dataset_name, f_score))

    if verbose: print("Writing results to %s" % output_dir)
    postprocess.write_results(results, output_dir, name="nmf-output.json")
    return results
