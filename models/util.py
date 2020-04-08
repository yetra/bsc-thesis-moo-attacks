import keras
from keras.datasets import mnist


def load_mnist(input_shape, num_of_outputs):
    """
    Loads the MNIST dataset.

    :param input_shape: a tuple describing the shape of the model's input layer
    :param num_of_outputs: the number of model outputs
    :return: the MNIST training & test sets with corresponding labels
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], *input_shape).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], *input_shape).astype('float32')

    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_of_outputs)
    y_test = keras.utils.to_categorical(y_test, num_of_outputs)

    return x_train, y_train, x_test, y_test
