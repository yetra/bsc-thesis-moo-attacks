import keras
from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
from matplotlib import pyplot as plt

INPUT_SIZE = 28 * 28
OUTPUT_SIZE = 10

LAYER_SIZES = [512, OUTPUT_SIZE]
ACTIVATION_FUNCTIONS = ['sigmoid', 'softmax']

BATCH_SIZE = 128
EPOCHS = 10


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


def plot_results(history):
    """
    Plots the model's accuracy over epochs on the training and validation sets.

    :param history: a History object containing loss and metrics values
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()


def main():
    """Builds, trains and displays the results of a model."""
    x_train, y_train, x_test, y_test = prepare_data(INPUT_SIZE, OUTPUT_SIZE)

    model = build_model(INPUT_SIZE, LAYER_SIZES, ACTIVATION_FUNCTIONS)
    model.summary()

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_split=.1, verbose=False)

    plot_results(history)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(f'\ntest loss: {loss:.3}, test accuracy: {accuracy:.3}')


if __name__ == '__main__':
    main()
