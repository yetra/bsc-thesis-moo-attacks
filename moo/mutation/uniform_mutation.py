import random

import numpy as np

from moo.mutation.mutation import Mutation


class UniformMutation(Mutation):
    """An implementation of uniform mutation.

    Solution variables chosen with the given probability are replaced with
    uniform random values selected between the MOO problem's upper and lower
    bounds.

    Attributes:
        probability: the probability of mutating a decision vector variable
        problem: the MOOP being optimized
    """

    def __init__(self, probability, problem):
        """Initializes UniformMutation attributes."""
        self.probability = probability
        self.problem = problem

    def mutate(self, solution):
        """Mutates the given solution."""
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                solution.variables[i] = random.uniform(self.problem.mins[i], # TODO
                                                       self.problem.maxs[i])

        np.clip(solution.variables, self.problem.mins, self.problem.maxs,
                out=solution.variables)
