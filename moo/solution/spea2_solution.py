from moo.solution.solution import Solution


class SPEA2Solution(Solution):
    """Models a SPEA2 solution.

        Attributes:
            dominators: a list of all solutions that dominate this solution
            strength: the number of solutions that this solution is dominates
            density: the inverse of the distance to the k-th nearest solution
            raw_fitness: the sum of the strengths of this solution's dominators
            fitness: a sum of the density and raw fitness of this solution
        """

    def __init__(self, problem, variables=None):
        """Initializes SPEA2Solution attributes.

        If the given decision variables vector is None, the variables vector
        will be initialized with random values.

        :param problem: an instance of Problem - the MOOP to optimize
        :param variables: the decision variables vector
        """
        super().__init__(problem, variables)

        self.dominators = []
        self.strength = 0

        self.density = -1
        self.raw_fitness = -1
        self.fitness = -1

    def __gt__(self, other):
        """Applies the fitness comparison operator (for minimization)."""
        return self.fitness < other.fitness
