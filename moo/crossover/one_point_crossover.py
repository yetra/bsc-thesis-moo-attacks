import random

from moo.crossover.crossover import Crossover


class OnePointCrossover(Crossover):
    """An implementation of one-point crossover.

    Attributes:
        problem: the MOOP being optimized
    """

    def __init__(self, problem):
        """Initializes ArithmeticCrossover attributes."""
        self.problem = problem

    def of(self, first_parent, second_parent):
        """Returns two child solutions obtained by arithmetic crossover."""
        point = random.randint(1, self.problem.variables_count - 1)

        variables = (first_parent.variables[:point]
                     + second_parent.variables[point:])
        first_child = first_parent.__class__(self.problem, variables)

        variables = (second_parent.variables[:point]
                     + first_parent.variables[point:])
        second_child = second_parent.__class__(self.problem, variables)

        return first_child, second_child

