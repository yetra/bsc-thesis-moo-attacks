import random


class Solution:
    """Models a multi-objective optimization problem's solution.

    Attributes:
        variables: the decision variables vector
        objectives: the objectives in a given decision variables vector
    """

    def __init__(self, problem=None):
        """Initializes Solution attributes.

        If problem is not None, the decision vector's values will
        be randomized.

        :param problem: an instance of Problem - the MOOP to optimize
        """
        self.variables = []
        self.objectives = []

        if problem:
            self.randomize(problem)

    def randomize(self, problem):
        """Randomizes the decision space variables of this solution.

        :param problem: an instance of Problem containing info on the dimension
                        of the decision space and variable constraints
        """
        for min_v, max_v in zip(problem.mins, problem.maxs):
            self.variables.append(random.uniform(min_v, max_v))

    def dominates(self, other):
        """Returns True if this solution dominates the given solution."""
        is_strictly_better = False

        for o1, o2 in zip(self.objectives, other.objectives):
            if o1 > o2:
                return False
            if o1 < o2:
                is_strictly_better = True

        return is_strictly_better
