from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, \
    MaxPooling2D
from keras.models import Sequential

import time
import util

MODEL_FILE = 'conv_model.json'
WEIGHTS_FILE = 'conv_weights.h5'

INPUT_SHAPE = (28, 28, 1)
OUTPUTS = 10

LAYER_SIZES = [32, 32, 64, 64, 128, 10]
ACTIVATION_FUNCTIONS = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']

BATCH_SIZE = 128
EPOCHS = 10


class ConvolutionalModel:
    """A convolutional model."""
    INPUT_SHAPE = (28, 28, 1)
    NUM_OUTPUTS = 10

    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adam'

    def __init__(self, layer_sizes=(32, 32, 64, 64, 128, 10),
                 activations=('relu', 'relu', 'relu', 'relu', 'relu', 'softmax')):
        """
        Inits SimpleModel attributes.

        :param layer_sizes: a tuple of hidden and output layer sizes
        :param activations: a tuple of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.activations = activations

        self.model = self._build()
        self.model.summary()

    def _build(self):
        """Builds this model."""
        model = Sequential()

        model.add(Conv2D(self.layer_sizes[0], 3, input_shape=self.INPUT_SHAPE))
        model.add(Activation(self.activations[0]))
        model.add(Conv2D(self.layer_sizes[1], 3))
        model.add(Activation(self.activations[1]))
        model.add(MaxPooling2D())

        model.add(Conv2D(self.layer_sizes[2], 3))
        model.add(Activation(self.activations[2]))
        model.add(Conv2D(self.layer_sizes[3], 3))
        model.add(Activation(self.activations[3]))
        model.add(MaxPooling2D())

        model.add(Flatten())

        model.add(Dense(self.layer_sizes[4]))
        model.add(Activation(self.activations[4]))
        model.add(Dropout(0.4))
        model.add(Dense(self.layer_sizes[5]))
        model.add(Activation(self.activations[5]))

        return model

    def train(self, data, epochs=40, batch_size=128):
        """
        Trains this model and saves the resulting weights.

        :param data: a tuple of training and test data with labels
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

        self.model.save_weights(f'conv_{time.strftime("%Y%m%d_%H%M%S")}.h5')

    def load_weights(self, weights_path):
        """Loads this model's weights and compiles the model."""
        self.model.load_weights(weights_path)
        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER,
                           metrics=['accuracy'])


def build_model(input_shape, layer_sizes, activation_functions):
    """
    Builds a convolutional model.

    :param input_shape: a tuple describing the shape of the input layer
    :param layer_sizes: a list of hidden layer & output layer sizes
    :param activation_functions: a list of activation functions for each layer
    :return: the convolutional model created from the given parameters
    """
    model = Sequential()

    model.add(Conv2D(layer_sizes[0], 3, input_shape=input_shape))
    model.add(Activation(activation_functions[0]))
    model.add(Conv2D(layer_sizes[1], 3))
    model.add(Activation(activation_functions[1]))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(layer_sizes[2], 3))
    model.add(Activation(activation_functions[2]))
    model.add(Conv2D(layer_sizes[3], 3))
    model.add(Activation(activation_functions[3]))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(layer_sizes[4]))
    model.add(Activation(activation_functions[4]))
    model.add(Dropout(0.4))
    model.add(Dense(layer_sizes[5]))
    model.add(Activation(activation_functions[5]))

    return model


def main():
    """Builds, trains and displays the results of a model."""
    x_train, y_train, x_test, y_test = util.load_mnist(INPUT_SHAPE, OUTPUTS)

    model = build_model(INPUT_SHAPE, LAYER_SIZES, ACTIVATION_FUNCTIONS)
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, validation_split=.1)

    util.plot_results(history)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
    print(f'\ntest loss: {loss:.3}, test accuracy: {accuracy:.3}')

    util.save(model, MODEL_FILE, WEIGHTS_FILE)


if __name__ == '__main__':
    main()
