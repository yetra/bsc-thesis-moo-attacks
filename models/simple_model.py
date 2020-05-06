from keras.layers import Dense, Activation
from keras.models import Sequential

import util
from attack_model import AttackModel


class SimpleModel(AttackModel):
    """A simple, non-convolutional model."""
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'sgd'

    INPUT_SHAPE = (28 * 28, )
    NUM_OUTPUTS = 10

    def __init__(self, layer_sizes=(128, 10), activations=('relu', 'softmax')):
        """
        Inits SimpleModel attributes.

        :param layer_sizes: a tuple of hidden and output layer sizes
        :param activations: a tuple of activation functions for each layer
        """
        super().__init__(layer_sizes, activations)

    def _build(self):
        """Builds this model."""
        model = Sequential()

        model.add(Dense(self.layer_sizes[0], input_shape=self.INPUT_SHAPE))
        model.add(Activation(self.activations[0]))

        for i, size in enumerate(self.layer_sizes[1:]):
            model.add(Dense(size))
            model.add(Activation(self.activations[i + 1]))

        return model


if __name__ == '__main__':
    data = util.load_mnist(SimpleModel.INPUT_SHAPE, SimpleModel.NUM_OUTPUTS)

    simple_model = SimpleModel()
    simple_model.train(data, epochs=40, batch_size=128)
    simple_model.save()
