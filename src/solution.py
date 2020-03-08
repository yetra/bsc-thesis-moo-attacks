import random


class Solution:
    """Models a multi-objective optimization problem's solution.

    Attributes:
        variables: the decision variables vector
        objectives: the objectives in a given decision variables vector
    """

    def __init__(self):
        """Initializes Solution attributes."""
        self.variables = []
        self.objectives = []

    def _randomize(self, problem):
        """Randomizes the decision space variables of this solution.

        Args:
            problem: the MOOP object containing info on the decision space dimension and constraints
        """
        for _ in range(problem.variables_count):
            self.variables.append(random.uniform(problem.mins, problem.maxs))
