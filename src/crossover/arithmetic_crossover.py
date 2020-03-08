from src.crossover.crossover import Crossover


class ArithmeticCrossover(Crossover):
    """An implementation of arithmetic crossover.

    Two parent decision space vectors are linearly combines to produce two new child solutions.

    Attributes:
        alpha: the weighting factor to use in the linear combination
    """

    def __init__(self, alpha):
        """Initializes ArithmeticCrossover attributes."""
        self.alpha = alpha

    def of(self, first_parent, second_parent):
        """Returns two child solutions obtained by arithmetically crossing the given parents."""
        pass
