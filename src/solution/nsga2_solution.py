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

    def __gt__(self, other):
        """Applies the crowding-comparison operator."""
        return self.rank < other.rank \
            or self.rank == other.rank \
            and self.crowding_distance > other.crowding_distance
