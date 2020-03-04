from abc import ABC, abstractmethod


class Crossover(ABC):
    """The base class to be extended by different crossover operators."""

    @abstractmethod
    def of(self, first_parent, second_parent):
        """Returns two child individuals obtained by crossing the given parents."""
