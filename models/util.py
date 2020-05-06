import keras
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
