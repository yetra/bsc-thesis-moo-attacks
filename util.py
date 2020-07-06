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


def sample_choice(x, y, labels, samples_per_label, seed=None):
    """
    Chooses samples_per_label random samples for each label in labels.

    :param x: the samples to choose from
    :param y: the corresponding labels
    :param labels: a collection of all possible labels
    :param samples_per_label: the number of samples to choose for each label
    :param seed: the seed to use for choosing random samples
    :return: num_samples random samples with the corresponding labels
    """
    rng = np.random if seed is None else np.random.RandomState(seed)
    rand_x, rand_y = [], []

    for label in labels:
        for _ in range(samples_per_label):
            index = rng.randint(0, len(x))
            while y[index] != label:
                index = rng.randint(0, len(x))

            rand_x.append(x[index])
            rand_y.append(label)

    return rand_x, rand_y


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
