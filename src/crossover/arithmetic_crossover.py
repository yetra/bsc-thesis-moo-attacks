from src.crossover.crossover import Crossover
from src.solution import Solution


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
        if len(first_parent.variables) != len(second_parent.variables):
            raise ValueError("Parent variable vectors must be of the same length!")

        first_child = Solution()
        second_child = Solution()

        for i in range(len(first_parent.variables)):
            fist_alpha = self.alpha * first_parent.variables[i]
            second_alpha = self.alpha * second_parent.variables[i]

            first_child.variables.append(fist_alpha + second_parent.variables[i] - second_alpha)
            second_child.variables.append(first_parent.variables[i] - fist_alpha + second_alpha)

        return first_child, second_child
