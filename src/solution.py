class Solution:
    """Models a multi-objective optimization problem's solution.

    Attributes:
        variables: the decision variables vector
        objectives: the objectives in a given decision variables vector
    """

    def __init__(self):
        """Initializes Solution attributes."""
        self.variables = []
        self.objectives = []
