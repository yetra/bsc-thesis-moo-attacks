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

        self.variables_count = 28 * 28
        self.objectives_count = 2

        self.mins = 0  # TODO
        self.maxs = 1

        self.objective_mins = [None, None]
        self.objective_maxs = [None, None]

    def evaluate(self, population, orig_image, label):
        """Evaluates solutions in the given population."""
        for solution in population:
            predictions = self.model.predict(orig_image + solution.variables)
            noise_strength = np.linalg.norm(solution.variables, ord=1)

            solution.objectives = np.array([predictions[label],
                                            noise_strength])

            for i, o in enumerate(solution.objectives):
                if self.objective_mins[i] is None or o < self.objective_mins[i]:
                    self.objective_mins[i] = o
                elif self.objective_maxs[i] is None or o > self.objective_maxs[i]:
                    self.objective_maxs[i] = o
