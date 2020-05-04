import numpy as np

from problem.problem import Problem


class SimpleAttack(Problem):
    """
    A MOO problem for simple, non-targeted attacks on image recognition models.
    """

    def __init__(self, model):
        """Initializes AttackProblem attributes."""
        super().__init__()

        self.model = model

        self.num_variables = 28 * 28
        self.num_objectives = 2

        self.mins = 0  # TODO
        self.maxs = 1

        self.o_mins = [None, None]
        self.o_maxs = [None, None]

    def evaluate(self, population, orig_image, label):
        """Evaluates solutions in the given population."""
        for solution in population:
            predictions = self.model.predict(orig_image + solution.variables)
            noise_strength = np.linalg.norm(solution.variables, ord=1)

            solution.objectives = np.array([predictions[label],
                                            noise_strength])

            for i, o in enumerate(solution.objectives):
                if self.o_mins[i] is None or o < self.o_mins[i]:
                    self.o_mins[i] = o
                elif self.o_maxs[i] is None or o > self.o_maxs[i]:
                    self.o_maxs[i] = o
