import random

from src.selection.selection import Selection


class TournamentSelection(Selection):
    """An implementation of tournament selection.

    The best solution is selected from a tournament of randomly chosen solutions
    from a given population.

    Attributes:
        tournament_size: the size of the tournament
    """

    def __init__(self, tournament_size):
        """Initializes TournamentSelection attributes."""
        self.tournament_size = tournament_size

    def select_from(self, population):
        """Returns a solution selected from the given population."""
        best_count = 0
        best = None

        while best_count < self.tournament_size:
            random_solution = random.choice(population)

            if best is None or random_solution > best:
                best = random_solution
                best_count += 1

        # TODO comparison operators - crowded tournament selection
