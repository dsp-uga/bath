import urllib.request
import argparse
import os
import sys
import zipfile

"""
Convenience script that downloads a customizable subset of the data with a really simple progress printout
The datasets are automatically unzipped in the output directory.
"""

description = 'CSCI 8630 Project 3 Data Downloader' \
              'This script downloads a subset of the neuroimage data to a local directory'

parser = argparse.ArgumentParser(description=description, add_help='How to use', prog='python download.py <options>')

# All args are optional - by default the first training example will be downloaded to the ./data directory
parser.add_argument("-d", "--datasets", default=["00.00"], nargs="+",
                    help="Datasets to download, separated by spaces [DEFAULT: \"00.00\"]")

parser.add_argument("-o", "--output", default="data",
                    help="Path to the directory where data will be downloaded. [DEFAULT: \"data/\"]")

args = parser.parse_args()

# should be file names like [00.00, 00.01, 00.00.test, etc]
datasets = args.datasets

# keep track of progress
total = len(datasets)
counter = 0
display_progress = lambda i: sys.stdout.write('\r%d%%' % (i*100 / total)); sys.stdout.flush()

# download each dataset and print progress
print("Starting...")
display_progress(0)
for dataset in datasets:
    filename = "neurofinder."+dataset+".zip"
    output_dir = args.output
    output_path = os.path.join(output_dir, filename)
    # ensure output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # path to raw image data in the dsp-uga bucket
    data_url = "https://storage.googleapis.com/uga-dsp/project3/" + filename

    # download data and unzip
    urllib.request.urlretrieve(data_url, output_path)
    with zipfile.ZipFile(output_path, 'r') as archive:
        archive.extractall(output_dir)
    os.remove(output_path)

    # display progress
    counter += 1
    display_progress(counter)

print("\nComplete!")
