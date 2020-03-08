from abc import ABC, abstractmethod


class Mutation(ABC):
    """The base class to be extended by different mutation operators."""

    @abstractmethod
    def mutate(self, solution):
        """Mutates the given solution."""
