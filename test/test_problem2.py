from src.problem import Problem


class TestProblem2(Problem):
    """Models the following multi-objective optimization problem:

     f1(x1, x2) = x1,
     f2(x1, x2) = (1 + x2) / x1
    """

    def __init__(self):
        """Initializes TestProblem2 attributes."""
        super().__init__()

        self.variables_count = 2
        self.objectives_count = 2

        self.mins = [0.1, 0.0]
        self.maxs = [1.0, 5.0]

        self.objective_mins = [None] * self.objectives_count
        self.objective_maxs = [None] * self.objectives_count

    def evaluate(self, solution):
        """Evaluates the given solution."""
        solution.objectives[0] = solution.variables[0]
        solution.objectives[1] = (1.0 + solution.variables[1]) / solution.variables[0]

        for i, o in enumerate(solution.objectives):
            if not self.objective_mins[i] or o < self.objective_mins[i]:
                self.objective_mins[i] = o
            elif not self.objective_maxs[i] or o > self.objective_maxs[i]:
                self.objective_maxs[i] = o
