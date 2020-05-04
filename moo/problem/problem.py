from abc import ABC, abstractmethod


class Problem(ABC):
    """Models a multi-objective optimization problem.

    Attributes:
        variables_count: the number of decision space variables that this
                         problem requires
        objectives_count: the number of objectives to optimize
        mins: the lowest possible values of each decision space variable
        maxs: the highest possible values of each decision space variable
        objective_mins: the lowest possible values of each objective
        objective_maxs: the highest possible values of each objective
    """

    def __init__(self):
        """Initializes Problem attributes."""
        self.variables_count = -1
        self.objectives_count = -1

        self.mins = []
        self.maxs = []

        self.objective_mins = []
        self.objective_maxs = []

    @abstractmethod
    def evaluate(self, population, orig_image, label):
        """Evaluates solutions in the given population."""
