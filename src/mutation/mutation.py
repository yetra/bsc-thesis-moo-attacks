from abc import ABC, abstractmethod


class Mutation(ABC):
    """The base class to be extended by different mutation operators."""

    def of(self, individual):
        """Returns a mutated copy of the given individual."""
        copied = individual.copy()
        self.mutate(copied)

        return copied

    @abstractmethod
    def mutate(self, individual):
        """Mutates the given individual."""
