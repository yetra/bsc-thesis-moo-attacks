import numpy as np

from problem.problem import Problem


class TestProblem2(Problem):
    """Models the following multi-objective optimization problem:

    f1(x1, x2) = x1,
    f2(x1, x2) = (1 + x2) / x1
    """
    NUM_VARIABLES = 2
    NUM_OBJECTIVES = 2
    MINS = [0.1, 0.0]
    MAXS = [1.0, 5.0]

    def __init__(self):
        """Initializes TestProblem2 attributes."""
        super().__init__(self.NUM_VARIABLES, self.NUM_OBJECTIVES,
                         self.MINS, self.MAXS)

    def evaluate(self, population, orig_image, label):
        """Evaluates the given solution."""
        for solution in population:
            x1, x2 = solution.variables
            solution.objectives = np.array([x1, (1.0 + x2) / x1])
            self._update_o_extremes(solution)
