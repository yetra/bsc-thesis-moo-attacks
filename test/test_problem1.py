import numpy as np

from problem.problem import Problem


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

        self.num_variables = 4
        self.num_objectives = 4

        self.mins = [-5.0] * self.num_variables
        self.maxs = [5.0] * self.num_variables

        self.o_mins = [None] * self.num_objectives
        self.o_maxs = [None] * self.num_objectives

    def evaluate(self, population, orig_image, label):
        """Evaluates the given solution."""
        for solution in population:
            objectives = []

            for i, v in enumerate(solution.variables):
                objective = v * v
                objectives.append(objective)

                if (self.o_mins[i] is None
                        or objective < self.o_mins[i]):
                    self.o_mins[i] = objective

                elif (self.o_maxs[i] is None
                      or objective > self.o_maxs[i]):
                    self.o_maxs[i] = objective

            solution.objectives = np.array(objectives)
