import numpy as np

from moo.problem import Problem


class TestProblem1(Problem):
    """Models the following multi-objective optimization problem:

    f1(x1, x2, x3, x4) = x1^2,
    f2(x1, x2, x3, x4) = x2^2,
    f3(x1, x2, x3, x4) = x3^2,
    f4(x1, x2, x3, x4) = x4^2
    """

    def __init__(self):
        """Initializes TestProblem1 attributes."""
        super().__init__()

        self.variables_count = 4
        self.objectives_count = 4

        self.mins = [-5.0] * self.variables_count
        self.maxs = [5.0] * self.variables_count

        self.objective_mins = [None] * self.objectives_count
        self.objective_maxs = [None] * self.objectives_count

    def evaluate(self, solution):
        """Evaluates the given solution."""
        objectives = []

        for i, v in enumerate(solution.variables):
            objective = v * v
            objectives.append(objective)

            if (self.objective_mins[i] is None
                    or objective < self.objective_mins[i]):
                self.objective_mins[i] = objective

            elif (self.objective_maxs[i] is None
                  or objective > self.objective_maxs[i]):
                self.objective_maxs[i] = objective

        solution.objectives = np.array(objectives)
