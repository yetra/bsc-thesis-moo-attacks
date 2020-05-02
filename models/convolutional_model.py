from keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, \
    MaxPooling2D
from keras.models import Sequential

import util
from attack_model import AttackModel


class ConvolutionalModel(AttackModel):
    """A convolutional model."""
    LOSS = 'categorical_crossentropy'
    OPTIMIZER = 'adam'

    INPUT_SHAPE = (28, 28, 1)
    NUM_OUTPUTS = 10

    def __init__(self, layer_sizes=(32, 32, 64, 64, 128, 10),
                 activations=('relu', 'relu', 'relu', 'relu', 'relu', 'softmax')):
        """
        Inits ConvolutionalModel attributes.

        :param layer_sizes: a tuple of hidden and output layer sizes
        :param activations: a tuple of activation functions for each layer
        """
        super().__init__(layer_sizes, activations)

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


if __name__ == '__main__':
    data = util.load_mnist(ConvolutionalModel.INPUT_SHAPE,
                           ConvolutionalModel.NUM_OUTPUTS)

    conv_model = ConvolutionalModel()
    conv_model.train(data, epochs=10, batch_size=128)
    conv_model.save()
