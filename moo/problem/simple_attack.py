import numpy as np

from moo.problem.problem import Problem


class SimpleAttack(Problem):
    """
    A MOO problem for simple, non-targeted attacks on image recognition models.
    """
    NUM_OBJECTIVES = 2

    def __init__(self, model, noise_size):
        """Initializes AttackProblem attributes."""
        super().__init__(model.INPUT_SHAPE, self.NUM_OBJECTIVES,
                         -noise_size, noise_size)

        self.model = model

    def evaluate(self, population, orig_image, label):
        """Evaluates solutions in the given population."""
        for solution in population:
            raw_adv_image = orig_image + solution.variables

            clip_delta = raw_adv_image - np.clip(raw_adv_image, 0, 1)
            solution.variables -= clip_delta

            predictions = self.model.predict(orig_image + solution.variables)
            noise_strength = np.sum(np.abs(solution.variables))

            solution.objectives = np.array([predictions[label], noise_strength])
            self._update_o_extremes(solution)
