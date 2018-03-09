# bath

Neuron Segmentation

Bath is an experiment for the [neurofinder](http://neurofinder.codeneuro.org/) challenge.
It was created over the course of three weeks for the Spring 2018 section of the
CSCI 8360 Data Science Practicum course at the University of Georgia.

## Approach

TODO

## Getting Started

These instructions will get you a copy of the project up and running on your local machine 
for development and testing purposes.

### Prerequisites

This project uses [Conda](https://conda.io/docs/) to manage dependencies.  
Conda is all you need to get started.

### Installing

The environment.yml file is used by Conda to create a virtual environment that includes all the project's dependencies (including Python!)

Navigate to the project directory and run the following command

`conda env create`

This will create a virtual environment named "bath". Activate the virtual environment with the following command

`source activate bath`

After the environment has been activated, the program can be run as follows:

`python main.py <options>`

To display program args and help, run

`python main.py -h`

## Getting the data
The `scripts/download.py` file can be used to download subsets of the neurofinder data.  Usage:

`python download.py -d <datasets> -o <directory>`

The `datasets` argument should be a space-separated list of names of the neurofinder datasets `00.00`, `00.01`, etc.
See the [NeuroFinder website](http://neurofinder.codeneuro.org/) for details on which datasets are available.

The `directory` argument specifies the directory on your local machine where the data will be downloaded.  
If the directory doesn't exist then it will be created.

The `scripts/download-full-dataset.sh` script can be used to download all the files used in this project.
Usage:

`./scripts/download-full-dataset.sh -o <directory>`

## Running the tests

TODO

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

TODO

## Built With

* [Python 3.6](https://www.python.org/)
* [Conda](https://conda.io/docs/)
* [Keras](https://keras.io/)

## Contributing

The `master` branch of this repo is write-protected.  Every pull request must pass a code review before being merged.
Other than that, there are no specific guidelines for contributing.
If you see something that can be improved, please send us a pull request!

## Authors

* [Raunak Dey](https://github.com/raun1)
* [Zach Jones](https://github.com/zachdj)
* [Niraj Kadam](https://github.com/nirajkadam)

See the [contributors file](CONTRIBUTORS.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* [Dr. Shannon Quinn](https://github.com/magsol) for running the excellent DSP course at UGA and providing project guidance
* [U-Net](https://arxiv.org/abs/1505.04597) - Neural Net architecture used in this project for segmentation
* [Code NeuroFinder](http://neurofinder.codeneuro.org/) - Providing the datasets and lots of example code

