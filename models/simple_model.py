from keras.layers import Dense, Activation
from keras.models import Sequential

import time
import util

MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'weights.h5'

INPUT_SHAPE = (28 * 28, )
OUTPUTS = 10

LAYER_SIZES = [128, 10]
ACTIVATION_FUNCTIONS = ['relu', 'softmax']

BATCH_SIZE = 128
EPOCHS = 40


class SimpleModel:
    """Models a simple, non-convolutional model."""
    INPUT_SHAPE = (28 * 28, )
    NUM_OUTPUTS = 10

    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'sgd'

    def __init__(self, layer_sizes=(128, 10), activations=('relu', 'softmax')):
        """
        Inits SimpleModel attributes.

        :param layer_sizes: a tuple of hidden layer & output layer sizes
        :param activations: a tuple of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.activations = activations

        self.model = self._build()
        self.model.summary()

    def _build(self):
        """Builds this model."""
        model = Sequential()

        model.add(Dense(self.layer_sizes[0], input_shape=self.INPUT_SHAPE))
        model.add(Activation(self.activations[0]))

        for i, size in enumerate(self.layer_sizes[1:]):
            model.add(Dense(size))
            model.add(Activation(self.activations[i + 1]))

        return model

    def train(self, data, epochs=40, batch_size=128):
        """
        Trains this model and saves the resulting weights.

        :param data: a tuple of training data and test data with labels
        :param epochs: the number of epochs
        :param batch_size: the batch size
        """
        x_train, y_train, x_test, y_test = data

        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER,
                           metrics=['accuracy'])

        history = self.model.fit(x_train, y_train, epochs=epochs,
                                 batch_size=batch_size, validation_split=.1)

        util.plot_results(history)

        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=False)
        print(f'\ntest loss: {loss:.3}, test accuracy: {accuracy:.3}')

        self.model.save_weights(f'simple_{time.strftime("%Y%m%d_%H%M%S")}.h5')

    def load_weights(self, weights_path):
        """Loads this model's weights and compiles the model."""
        self.model.load_weights(weights_path)
        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER,
                           metrics=['accuracy'])


def build_model(input_shape, layer_sizes, activation_functions):
    """
    Builds a simple, non-convolutional model.

    :param input_shape: a tuple describing the shape of the input layer
    :param layer_sizes: a list of hidden layer & output layer sizes
    :param activation_functions: a list of activation functions for each layer
    :return: the model created from the given parameters
    """
    model = Sequential()

    model.add(Dense(layer_sizes[0], input_shape=input_shape))
    model.add(Activation(activation_functions[0]))

    for i, size in enumerate(layer_sizes[1:]):
        model.add(Dense(size))
        model.add(Activation(activation_functions[i + 1]))

    return model


def main():
    """Builds, trains and displays the results of a model."""
    x_train, y_train, x_test, y_test = util.load_mnist(INPUT_SHAPE, OUTPUTS)

    model = build_model(INPUT_SHAPE, LAYER_SIZES, ACTIVATION_FUNCTIONS)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, validation_split=.1)

    util.plot_results(history)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(f'\ntest loss: {loss:.3}, test accuracy: {accuracy:.3}')

    util.save(model, MODEL_FILE, WEIGHTS_FILE)


if __name__ == '__main__':
    main()
