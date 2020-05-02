from abc import ABC, abstractmethod

import time
from typing import ClassVar

import numpy as np

import util


class AttackModel(ABC):
    """The base class for attack models."""
    LOSS: ClassVar[str]
    OPTIMIZER: ClassVar[str]

    def __init__(self, layer_sizes, activations):
        """
        Inits AttackModel attributes.

        :param layer_sizes: a tuple of hidden layer & output layer sizes
        :param activations: a tuple of activation functions for each layer
        """
        self.layer_sizes = layer_sizes
        self.activations = activations

        self.model = self._build()
        self.model.summary()

    @abstractmethod
    def _build(self):
        """Builds this model."""

    def train(self, data, epochs, batch_size):
        """
        Trains this model.

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

    def predict(self, data):
        """Returns the predictions obtained for the given data."""
        return self.model.predict(np.array([data]))[0]

    def save(self, weights_path=None):
        """Saves this model's weights to the specified path."""
        if weights_path is None:
            weights_path = (f'data/{type(self).__name__}'
                            f'_{time.strftime("%Y%m%d_%H%M%S")}.h5')

        self.model.save_weights(weights_path)

    def load(self, weights_path):
        """Compiles this model using weights loaded from the specified path."""
        self.model.load_weights(weights_path)
        self.model.compile(loss=self.LOSS, optimizer=self.OPTIMIZER,
                           metrics=['accuracy'])
