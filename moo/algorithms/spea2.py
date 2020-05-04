import numpy as np

from solution.spea2_solution import SPEA2Solution


class SPEA2:
    """An implementation of SPEA-II.

    Zitzler, Eckart, Marco Laumanns, and Lothar Thiele. "SPEA2: Improving the
    strength Pareto evolutionary algorithm." TIK-report 103 (2001).

    Attributes:
        problem: the multi-objective optimization problem to solve
        population_size: the size of the population
        archive_size: the size of the archive
        max_iterations: the maximum number of algorithm iterations
        crossover: the crossover operator to use
        mutation: the mutation operator to use
        selection: the selection operator to use
    """

    def __init__(self, problem, population_size, archive_size, max_iterations,
                 crossover, mutation, selection):
        """Initializes SPEA2 attributes."""
        self.problem = problem

        self.population_size = population_size
        self.archive_size = archive_size
        self.max_iterations = max_iterations

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

    def run(self, orig_image, label):
        """Executes the algorithm."""
        population = self.generate_initial_population()
        self.problem.evaluate(population, orig_image, label)
        archive = []

        iteration = 0
        while iteration < self.max_iterations:
            print(f'i={iteration}')

            union = population + archive
            self.fitness_assignment(union)

            archive = self.environmental_selection(union)
            population = self.generate_next_population(archive)
            self.problem.evaluate(population, orig_image, label)
            
            iteration += 1

        return archive

    def generate_initial_population(self):
        """Returns the initial population."""
        population = []

        for _ in range(self.population_size):
            solution = SPEA2Solution(self.problem)
            population.append(solution)

        return population

    def generate_next_population(self, parents):
        """Generates population_size offspring from the given parents."""
        offspring = []

        while len(offspring) < self.population_size:
            first_parent = self.selection.select_from(parents)
            second_parent = self.selection.select_from(parents)
            children = self.crossover.of(first_parent, second_parent)

            for child in children:
                self.mutation.mutate(child)
                offspring.append(child)

        return offspring

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
                solution.density = distances[k]  # TODO add separate attribute

            archive.sort(key=lambda s: s.density, reverse=True)
            archive.pop()
