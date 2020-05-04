from abc import ABC, abstractmethod


class Problem(ABC):
    """Models a multi-objective optimization problem.

    Attributes:
        num_variables: the number of decision space variables that this
                         problem requires
        num_objectives: the number of objectives to optimize
        mins: the lowest possible values of each decision space variable
        maxs: the highest possible values of each decision space variable
        o_mins: the lowest possible values of each objective
        o_mins: the highest possible values of each objective
    """

    def __init__(self):
        """Initializes Problem attributes."""
        self.num_variables = -1
        self.num_objectives = -1

        self.mins = []
        self.maxs = []

        self.o_mins = []
        self.o_mins = []

    @abstractmethod
    def evaluate(self, population, orig_image, label):
        """Evaluates solutions in the given population."""
