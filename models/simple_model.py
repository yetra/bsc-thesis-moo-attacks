from keras.layers import Dense, Activation
from keras.models import Sequential

import time
import util


class SimpleModel:
    """Models a simple, non-convolutional model."""
    INPUT_SHAPE = (28 * 28, )
    NUM_OUTPUTS = 10

    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'sgd'

    def __init__(self, layer_sizes=(128, 10), activations=('relu', 'softmax')):
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

        model.add(Dense(self.layer_sizes[0], input_shape=self.INPUT_SHAPE))
        model.add(Activation(self.activations[0]))

        for i, size in enumerate(self.layer_sizes[1:]):
            model.add(Dense(size))
            model.add(Activation(self.activations[i + 1]))

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

        self.model.save_weights(f'simple_{time.strftime("%Y%m%d_%H%M%S")}.h5')

    def load_weights(self, weights_path):
        """Loads this model's weights and compiles the model."""
        self.model.load_weights(weights_path)
        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER,
                           metrics=['accuracy'])


if __name__ == '__main__':
    data = util.load_mnist(SimpleModel.INPUT_SHAPE, SimpleModel.NUM_OUTPUTS)

    simple_model = SimpleModel()
    simple_model.train(data)
