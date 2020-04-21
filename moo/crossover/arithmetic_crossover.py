from moo.crossover.crossover import Crossover


class ArithmeticCrossover(Crossover):
    """An implementation of arithmetic crossover.

    Two parent decision space vectors are linearly combines to produce
    two new child solutions.

    Attributes:
        alpha: the weighting factor to use in the linear combination
        problem: the MOOP being optimized
    """

    def __init__(self, alpha, problem):
        """Initializes ArithmeticCrossover attributes."""
        self.alpha = alpha
        self.problem = problem

    def of(self, first_parent, second_parent):
        """Returns two child solutions obtained by arithmetic crossover."""
        variables = (self.alpha * first_parent.variables
                     + (1 - self.alpha) * second_parent.variables)
        first_child = first_parent.__class__(first_parent.problem, variables)

        variables = ((1 - self.alpha) * first_parent.variables
                     + self.alpha * second_parent.variables)
        second_child = second_parent.__class__(first_parent.problem, variables)

        self.problem.check_constraints(first_child)
        self.problem.check_constraints(second_child)

        return first_child, second_child
