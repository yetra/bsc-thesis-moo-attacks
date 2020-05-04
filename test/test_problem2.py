import numpy as np

from problem.problem import Problem


class TestProblem2(Problem):
    """Models the following multi-objective optimization problem:

    f1(x1, x2) = x1,
    f2(x1, x2) = (1 + x2) / x1
    """

    def __init__(self):
        """Initializes TestProblem2 attributes."""
        super().__init__()

        self.num_variables = 2
        self.num_objectives = 2

        self.mins = [0.1, 0.0]
        self.maxs = [1.0, 5.0]

        self.o_mins = [None] * self.num_objectives
        self.o_maxs = [None] * self.num_objectives

    def evaluate(self, population, orig_image, label):
        """Evaluates the given solution."""
        for solution in population:
            solution.objectives = np.array(
                [solution.variables[0],
                 (1.0 + solution.variables[1]) / solution.variables[0]]
            )

            for i, o in enumerate(solution.objectives):
                if self.o_mins[i] is None or o < self.o_mins[i]:
                    self.o_mins[i] = o
                elif self.o_maxs[i] is None or o > self.o_maxs[i]:
                    self.o_maxs[i] = o
