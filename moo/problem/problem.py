from abc import ABC, abstractmethod


class Problem(ABC):
    """Models a multi-objective optimization problem.

    Attributes:
        num_variables: the number of decision space variables
        num_objectives: the number of objectives to optimize
        mins: the lowest possible values of each decision space variable
        maxs: the highest possible values of each decision space variable
        o_mins: the lowest possible values of each objective
        o_mins: the highest possible values of each objective
    """

    def __init__(self, num_variables, num_objectives, mins, maxs,
                 o_mins=None, o_maxs=None):
        """Initializes Problem attributes."""
        self.num_variables = num_variables
        self.num_objectives = num_objectives

        self.mins = mins
        self.maxs = maxs

        self.o_mins = ([float('inf')] * num_objectives
                       if o_mins is None else o_mins)
        self.o_maxs = ([float('-inf')] * num_objectives
                       if o_maxs is None else o_maxs)

    def _update_o_extremes(self, solution):
        """Updates the o_mins & o_maxs attributes."""
        for i, o in enumerate(solution.objectives):
            if o < self.o_mins[i]:
                self.o_mins[i] = o
            elif o > self.o_maxs[i]:
                self.o_maxs[i] = o

    @abstractmethod
    def evaluate(self, population, orig_image, label):
        """Evaluates solutions in the given population."""
