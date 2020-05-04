import numpy as np

from problem.problem import Problem


class TargetedAttack(Problem):
    """A MOO problem for targeted attacks on image recognition models."""
    NUM_VARIABLES = 28 * 28
    NUM_OBJECTIVES = 2

    def __init__(self, model, noise_size):
        """Initializes TargetedAttack attributes."""
        super().__init__(self.NUM_VARIABLES, self.NUM_OBJECTIVES,
                         -noise_size, noise_size)

        self.model = model

    def evaluate(self, population, orig_image, label):
        """Evaluates the given solution."""
        for solution in population:
            predictions = self.model.predict(orig_image + solution.variables)
            noise_strength = np.linalg.norm(solution.variables, ord=1)

            solution.objectives = np.array([-predictions[label], noise_strength])
            self._update_o_extremes(solution)
