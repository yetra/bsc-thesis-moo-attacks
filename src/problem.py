from abc import ABC, abstractmethod


class Problem(ABC):
    """Models a multi-objective optimization problem.

    Attributes:
        variables_count: the number of decision space variables that this problem requires
        objectives_count: the number of objectives to optimize
        mins: the lowest possible values of each decision space variable
        maxs: the highest possible values of each decision space variable
        objective_mins: the lowest possible values of each objective
        objective_maxs: the highest possible values of each objective
    """

    def __init__(self, variables_count, objectives_count, mins, maxs, objective_mins, objective_maxs):
        """Initializes Problem attributes."""
        self.variables_count = variables_count
        self.objectives_count = objectives_count

        self.mins = mins
        self.maxs = maxs

        self.objective_mins = objective_mins
        self.objective_maxs = objective_maxs

    @abstractmethod
    def evaluate(self, solution):
        """Evaluates the given solution."""
