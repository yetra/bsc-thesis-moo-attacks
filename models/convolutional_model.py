from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, \
    MaxPooling2D
from keras.models import Sequential

INPUT_SHAPE = (28, 28, 1)
OUTPUTS = 10

LAYER_SIZES = [32, 32, 64, 64, 128, 10]
ACTIVATION_FUNCTIONS = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']

BATCH_SIZE = 128
EPOCHS = 10


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
