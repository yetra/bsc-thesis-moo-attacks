# bsc-thesis-moo-attacks

## Getting Started

Download or clone the [bsc-thesis-moo-attacks](https://github.com/yetra/bsc-thesis-moo-attacks) repository.

### Prerequisites

* Python 3.7+ - [Download & install](https://www.python.org/downloads/) the latest version.
* Keras - Easiest to install with `pip`. [Getting started guide](https://keras.io/getting_started/).
* NumPy - Installation instructions [here](https://numpy.org/install/).
* Matplotlib - Installation instructions [here](https://matplotlib.org/users/installing.html).
* Pillow - Installation instructions [here](https://pillow.readthedocs.io/en/stable/installation.html).

## Generating adversarial examples

On the command-line, `cd` to the `bsc-thesis-moo-attacks` directory. 

The main program that builds adversarial examples for 30 random MNIST samples can be run with: 
```shell script
$ python3 main.py algorithm attack_type noise_size model weights [-h] [--maxiter MAXITER] [--popsize POPSIZE]
```

The following command-line arguments are required:
* `algorithm` - the MOO algorithm to run [`nsga2` or `spea2`]
* `attack_type` - the type of adversarial attack to use [`simple_attack`, `targeted_attack` or `improved_attack`]
* `noise_size` - the size of the noise [real number in range 0-1]
* `model` - the classification model to attack [`simple_model` or `convolutional_model`]
* `weights` - path to the weights file

The optional arguments `--maxiter` and `--popsize` can be used to specify the maximum number of algorithm iterations and population size, respectively.

Alternatively, open the `bsc-thesis-moo-attacks` directory in an IDE (such as [PyCharm](https://www.jetbrains.com/pycharm/)) and run the program from there.
