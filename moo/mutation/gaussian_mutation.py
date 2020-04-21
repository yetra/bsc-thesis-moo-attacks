import random

import numpy as np

from moo.mutation.mutation import Mutation


class GaussianMutation(Mutation):
    """An implementation of Gaussian mutation.

    A Gaussian distributed random value is added to a solution's decision
    vector variables with the specified probability.

    Attributes:
        mu: the mean
        sigma: the standard deviation
        probability: the probability of mutating a decision vector variable
        problem: the MOOP being optimized
    """

    def __init__(self, mu, sigma, probability, problem):
        """Initializes GaussianMutation attributes."""
        self.mu = mu
        self.sigma = sigma
        self.probability = probability
        self.problem = problem

    def mutate(self, solution):
        """Mutates the given solution."""
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                solution.variables[i] += random.gauss(self.mu, self.sigma)

        np.clip(solution.variables, self.problem.mins, self.problem.maxs,
                out=solution.variables)
