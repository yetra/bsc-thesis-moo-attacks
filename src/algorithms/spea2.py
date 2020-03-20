import math

from solution.solution import Solution


class SPEA2:
    """An implementation of SPEA-II.

    Zitzler, Eckart, Marco Laumanns, and Lothar Thiele. "SPEA2: Improving the
    strength Pareto evolutionary algorithm." TIK-report 103 (2001).

    Attributes:
        problem: the multi-objective optimization problem to solve
        population_size: the size of the population
        max_iterations: the maximum number of algorithm iterations
        crossover: the crossover operator to use
        mutation: the mutation operator to use
        selection: the selection operator to use
    """

    def __init__(self, problem, population_size, max_iterations, crossover,
                 mutation, selection):
        """Initializes SPEA2 attributes."""
        self.problem = problem

        self.population_size = population_size
        self.max_iterations = max_iterations

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

    def run(self):
        """Executes the algorithm."""
        population = self.generate_initial_population()
        archive = []

        iteration = 0
        while iteration < self.max_iterations:
            union = population + archive
            k = int(math.sqrt(len(union)))

            for solution in union:
                distances = []

                for candidate in union:
                    distance = self.euclidean_distance(solution, candidate)
                    distances.append(distance)

                    if solution.dominates(candidate):
                        candidate.dominators.append(solution)
                        solution.strength += 1

                distances.sort()
                solution.density = 1.0 / (distances[k] + 2.0)

            for solution in union:
                solution.raw_fitness = sum(d.strength for d in solution.dominators)
                solution.fitness = solution.raw_fitness + solution.density

            next_archive = [s for s in union if s.fitness < 1]

            if len(next_archive) < len(archive):
                union.sort(key=lambda s: s.fitness, reverse=True)
                fill_count = len(archive) - len(next_archive)
                next_archive += union[:fill_count]

            elif len(next_archive) > len(archive):  # TODO iterative truncation
                for solution in next_archive:
                    distances = []

                    for other in next_archive:
                        distance = self.euclidean_distance(solution, other)
                        distances.append(distance)

                    distances.sort()
                    solution.kth_distance = distances[k]

                next_archive.sort()  # TODO implement comparison operator

                remove_count = len(next_archive) - len(archive)
                next_archive = next_archive[remove_count:]

    def generate_initial_population(self):
        """Returns the initial population."""
        population = []

        for _ in range(self.population_size):
            solution = Solution(self.problem)
            self.problem.evaluate(solution)
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
                self.problem.evaluate(child)
                offspring.append(child)

        return offspring

    def euclidean_distance(self, first_solution, second_solution):
        """Returns the Euclidean distance between the given solutions."""
        distance = 0.0

        for o1, o2 in zip(first_solution.objectives, second_solution.objectives):
            distance += (o2 - o1) ** 2

        return math.sqrt(distance)
