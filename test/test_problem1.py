import numpy as np

from problem.problem import Problem


class TestProblem1(Problem):
    """Models the following multi-objective optimization problem:

    f1(x1, x2, x3, x4) = x1^2,
    f2(x1, x2, x3, x4) = x2^2,
    f3(x1, x2, x3, x4) = x3^2,
    f4(x1, x2, x3, x4) = x4^2
    """
    NUM_VARIABLES = 4
    NUM_OBJECTIVES = 4
    MINS = [-5.0, -5.0, -5.0, -5.0]
    MAXS = [5.0, 5.0, 5.0, 5.0]

    def __init__(self):
        """Initializes TestProblem1 attributes."""
        super().__init__(self.NUM_VARIABLES, self.NUM_OBJECTIVES,
                         self.MINS, self.MAXS)

    def evaluate(self, population, orig_image, label):
        """Evaluates the given solution."""
        for solution in population:
            solution.objectives = np.array([v * v for v in solution.variables])
            self._update_o_extremes(solution)
