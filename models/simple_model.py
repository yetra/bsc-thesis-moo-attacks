import keras
from keras.datasets import mnist


def prepare_data(input_size, output_size):
    """Prepares the MNIST dataset for training."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], input_size)
    x_test = x_test.reshape(x_test.shape[0], input_size)

    y_train = keras.utils.to_categorical(y_train, output_size)
    y_test = keras.utils.to_categorical(y_test, output_size)

    return x_train, y_train, x_test, y_test
