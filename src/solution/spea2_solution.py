import math

from solution.solution import Solution


class SPEA2Solution(Solution):
    """Models a SPEA2 solution.

        Attributes:
            dominators: a list of all solutions that dominate this solution
            strength: the number of solutions that this solution is dominates
            density: the inverse of the distance to the k-th nearest solution
            raw_fitness: the sum of the strengths of this solution's dominators
            fitness: a sum of the density and raw fitness of this solution
        """

    def __init__(self, problem=None):
        """Initializes Solution attributes.

        If the problem argument is not None, the decision vector's values will
        be randomized.

        Args:
            problem: an object representing the MOOP to optimize
        """
        super().__init__(problem)

        self.dominators = []
        self.strength = -1

        self.density = -1
        self.raw_fitness = -1
        self.fitness = -1

    def __gt__(self, other):
        """Applies the fitness comparison operator (for minimization)."""
        return self.fitness < other.fitness

    def euclidean_distance(self, other):
        """Returns the objective-space Euclidean distance to the given other
        solution."""
        distance = 0.0

        for o1, o2 in zip(self.objectives, other.objectives):
            distance += (o2 - o1) ** 2

        return math.sqrt(distance)
