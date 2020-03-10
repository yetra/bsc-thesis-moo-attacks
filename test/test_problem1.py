from src.problem import Problem


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
        for i, v, o in enumerate(zip(solution.variables, solution.objectives)):
            solution.objectives[i] = v * v

            if not self.objective_mins[i] or o < self.objective_mins[i]:
                self.objective_mins[i] = o
            elif not self.objective_maxs[i] or o > self.objective_maxs[i]:
                self.objective_maxs[i] = o
