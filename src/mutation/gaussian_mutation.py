import random

from src.mutation.mutation import Mutation


class GaussianMutation(Mutation):
    """An implementation of Gaussian mutation.

    A Gaussian distributed random value is added to a solution's decision vector variables
    with the specified probability.

    Attributes:
        mu: the mean
        sigma: the standard deviation
        probability: the probability of mutating a decision vector variable

    """

    def __init__(self, mu, sigma, probability):
        """Initializes GaussianMutation attributes."""
        self.mu = mu
        self.sigma = sigma
        self.probability = probability

    def mutate(self, solution):
        """Mutates the given solution."""
        for i in range(len(solution.variables)):
            if random.random() <= self.probability:
                solution.variables[i] += random.gauss(self.mu, self.sigma)
