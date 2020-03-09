from src.solution.solution import Solution


class NSGA2Solution(Solution):
    """Models an NSGA2 solution.

    Attributes:
        dominates_list: a list of all solutions that this solution dominates
        dominated_by: the number of solutions that this solution is dominated by
        crowding_distance: the crowding distance value of this solution
        rank: the rank of this solution
    """
    def __init__(self, problem=None):
        """Initializes Solution attributes.

        If the problem argument is not None, the decision vector's values will be randomized.

        Args:
            problem: an object representing the MOOP to optimize
        """
        super().__init__(problem)

        self.dominates_list = []
        self.dominated_by = -1

        self.crowding_distance = -1
        self.rank = -1

    def dominates(self, other):
        """Returns True if this solution dominates the given solution."""
        is_strictly_better = False

        for o1, o2 in zip(self.objectives, other.objectives):
            if o1 > o2:
                return False
            if o1 < o2:
                is_strictly_better = True

        return is_strictly_better
