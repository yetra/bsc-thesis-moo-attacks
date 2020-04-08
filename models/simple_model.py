import keras
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
from matplotlib import pyplot as plt


def prepare_data(input_size, output_size):
    """Prepares the MNIST dataset for training."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], input_size)
    x_test = x_test.reshape(x_test.shape[0], input_size)

    y_train = keras.utils.to_categorical(y_train, output_size)
    y_test = keras.utils.to_categorical(y_test, output_size)

    return x_train, y_train, x_test, y_test


def build_model(input_size, layer_sizes, activation_functions):
    """
    Builds a simple, non-convolutional model.

    :param input_size: the number of inputs
    :param layer_sizes: a list of hidden layer & output layer sizes
    :param activation_functions: a list of activation functions for each layer
    :return: the model created from the given parameters
    """
    model = Sequential()

    model.add(Dense(layer_sizes[0], input_dim=input_size))
    model.add(Activation(activation_functions[0]))

    for i, size in enumerate(layer_sizes[1:]):
        model.add(Dense(size))
        model.add(Activation(activation_functions[i + 1]))

    return model
