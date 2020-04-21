import numpy as np


class Solution:
    """Models a multi-objective optimization problem's solution.

    Attributes:
        problem: the multi-objective optimization problem
        variables: the decision variables vector
        objectives: the objectives in a given decision variables vector
    """

    def __init__(self, problem, variables=None):
        """Initializes Solution attributes.

        If the given decision variables vector is None, the variables vector
        will be initialized with random values.

        :param problem: an instance of Problem - the MOOP to optimize
        :param variables: the decision variables vector
        """
        self.problem = problem

        if variables is None:
            self.variables = np.random.uniform(problem.mins, problem.maxs)
        else:
            self.variables = variables

        self.objectives = np.zeros(problem.objectives_count)

    def dominates(self, other):
        """Returns True if this solution dominates the given solution."""
        is_strictly_better = False

        for o1, o2 in np.nditer([self.objectives, other.objectives]):
            if o1 > o2:
                return False
            if o1 < o2:
                is_strictly_better = True

        return is_strictly_better
