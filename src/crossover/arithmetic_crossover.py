from src.crossover.crossover import Crossover
from src.solution.solution import Solution


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
        first_child, second_child = first_parent.__class__(), second_parent.__class__()

        for v1, v2 in zip(first_parent.variables, second_parent.variables):
            first_child.variables.append(self.alpha * v1 + (1 - self.alpha) * v2)
            second_child.variables.append((1 - self.alpha) * v1 + self.alpha * v2)

        return first_child, second_child
