import numpy as np

from moo.problem.problem import Problem


class ImprovedTargetedAttack(Problem):
    """A MOO problem for targeted attacks on image recognition models."""
    NUM_OBJECTIVES = 2

    def __init__(self, model, noise_size):
        """Initializes TargetedAttack attributes."""
        super().__init__(model.INPUT_SHAPE, self.NUM_OBJECTIVES,
                         -noise_size, noise_size)

        self.model = model

    def evaluate(self, population, orig_image, label):
        """Evaluates the given solution."""
        for solution in population:
            predictions = self.model.predict(orig_image + solution.variables)
            noise_strength = np.sum(np.abs(solution.variables))

            solution.objectives = np.array([-predictions[label],
                                            1.0 - predictions[label],
                                            noise_strength])

            self._update_o_extremes(solution)
