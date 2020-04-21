from moo.solution.solution import Solution


class NSGA2Solution(Solution):
    """Models an NSGA2 solution.

    Attributes:
        dominates_list: a list of all solutions that this solution dominates
        dominated_by: the number of solutions that dominate this solution
        crowding_distance: the crowding distance value of this solution
        rank: the rank of this solution
    """

    def __init__(self, problem, variables=None):
        """Initializes NSGA2Solution attributes.

        If the given decision variables vector is None, the variables vector
        will be initialized with random values.

        :param problem: an instance of Problem - the MOOP to optimize
        :param variables: the decision variables vector
        """
        super().__init__(problem, variables)

        self.dominates_list = []
        self.dominated_by = -1

        self.crowding_distance = -1
        self.rank = -1

    def __gt__(self, other):
        """Applies the crowding-comparison operator."""
        return self.rank < other.rank \
            or self.rank == other.rank \
            and self.crowding_distance > other.crowding_distance
