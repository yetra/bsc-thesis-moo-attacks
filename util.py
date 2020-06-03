import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt


def load_mnist(input_shape, num_outputs=10):
    """
    Loads the MNIST dataset.

    :param input_shape: a tuple describing the shape of the model's input layer
    :param num_outputs: the number of model outputs
    :return: the MNIST training & test sets with corresponding labels
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], *input_shape).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], *input_shape).astype('float32')

    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, num_outputs)
    y_test = keras.utils.to_categorical(y_test, num_outputs)

    return x_train, y_train, x_test, y_test


def mnist_choice(data, num_images, test_split=0.5, seed=None):
    """
    Chooses num_images random MNIST images.

    :param data: a tuple of train and test data to choose from
    :param num_images: the number of random images to choose
    :param test_split: the percentage of test images in the random choice
    :param seed: the seed to use for choosing random images
    :return: num_images random MNIST images with corresponding labels
    """
    x_train, y_train, x_test, y_test = data

    num_train = int(num_images * test_split)
    num_test = num_images - num_train

    rng = np if seed is None else np.random.RandomState(seed)
    i_train = rng.random.choice(len(x_train), size=num_train, replace=False)
    i_test = rng.random.choice(len(x_test), size=num_test, replace=False)

    images = np.append(x_train[i_train], x_test[i_test], axis=0)
    labels = np.append(y_train[i_train], y_test[i_test], axis=0)

    return images, labels


def plot_results(history):
    """
    Plots a model's accuracy & loss values for the training & validation sets.

    :param history: a History object containing loss and metrics values
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()
