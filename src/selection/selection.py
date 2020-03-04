from abc import ABC, abstractmethod


class Selection(ABC):
    """The base class to be extended by different selection operators."""

    @abstractmethod
    def select_from(self, population):
        """Returns an individual selected from the given population."""
