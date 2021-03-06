import numpy as np

from moo import operators
from moo.solution.spea2_solution import SPEA2Solution


class SPEA2:
    """An implementation of SPEA-II.

    Zitzler, Eckart, Marco Laumanns, and Lothar Thiele. "SPEA2: Improving the
    strength Pareto evolutionary algorithm." TIK-report 103 (2001).

    Attributes:
        problem: the multi-objective optimization problem to solve
        pop_size: the size of the population
        archive_size: the size of the archive
        max_iterations: the maximum number of algorithm iterations
    """

    def __init__(self, problem, pop_size, archive_size, max_iterations):
        """Initializes SPEA2 attributes."""
        self.problem = problem

        self.pop_size = pop_size
        self.archive_size = archive_size
        self.max_iterations = max_iterations

    def run(self, orig_image, label):
        """Executes the algorithm."""
        population = self.initialize()
        archive = []

        iteration = 0
        while iteration < self.max_iterations:
            # print(f'i={iteration}')

            self.problem.evaluate(population, orig_image, label)

            union = population + archive
            self.fitness_assignment(union)

            archive = self.environmental_selection(union)
            population = operators.reproduce(archive)

            iteration += 1

        nondominated = [s for s in archive if s.fitness < 1]

        return nondominated

    def initialize(self):
        """Returns the initial population."""
        return [SPEA2Solution(self.problem) for _ in range(self.pop_size)]

    def fitness_assignment(self, union):
        """Assigns fitness values to all solutions in the given union."""
        k = int(np.sqrt(len(union)))

        for solution in union:
            solution.strength = 0
            distances = []

            for candidate in union:
                distances.append(np.linalg.norm(
                    solution.objectives - candidate.objectives))

                if solution.dominates(candidate):
                    candidate.dominators.append(solution)
                    solution.strength += 1

            distances.sort()
            solution.density = 1.0 / (distances[k] + 2.0)

        for solution in union:
            solution.raw_fitness = sum(d.strength for d in solution.dominators)
            solution.fitness = solution.raw_fitness + solution.density
            solution.dominators = []

    def environmental_selection(self, union):
        """Applies environmental selection to the given union and returns
        the obtained new archive."""
        next_archive = [s for s in union if s.fitness < 1]

        if len(next_archive) < self.archive_size:
            union.sort(reverse=True)
            
            for solution in union:
                if solution.fitness >= 1:
                    next_archive.append(solution)

                if len(next_archive) >= self.archive_size:
                    break

        elif len(next_archive) > self.archive_size:
            self.archive_truncation(next_archive)

        return next_archive

    def archive_truncation(self, archive):
        """Truncates the given archive to archive_size."""
        while len(archive) > self.archive_size:
            k = int(np.sqrt(len(archive)))

            for solution in archive:
                distances = []

                for other in archive:
                    distances.append(np.linalg.norm(
                        solution.objectives - other.objectives))

                distances.sort()
                solution.density = distances[k]

            archive.sort(key=lambda s: s.density, reverse=True)
            archive.pop()
