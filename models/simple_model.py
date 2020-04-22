from keras.layers import Dense, Activation
from keras.models import Sequential

import util

MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'weights.h5'

INPUT_SHAPE = (28 * 28, )
OUTPUTS = 10

LAYER_SIZES = [128, 10]
ACTIVATION_FUNCTIONS = ['relu', 'softmax']

BATCH_SIZE = 128
EPOCHS = 40


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
