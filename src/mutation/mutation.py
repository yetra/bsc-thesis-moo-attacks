from abc import ABC, abstractmethod


class Mutation(ABC):
    """The base class to be extended by different mutation operators."""

    def of(self, solution):
        """Returns a mutated copy of the given solution."""
        copied = solution.copy()
        self.mutate(copied)

        return copied

    @abstractmethod
    def mutate(self, solution):
        """Mutates the given solution."""
