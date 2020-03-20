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

            for solution in union:
                for candidate in union:
                    if solution.dominates(candidate):
                        candidate.dominators.append(solution)
                        solution.strength += 1

            for solution in union:
                solution.raw_fitness = sum(d.strength for d in solution.dominators)

    def generate_initial_population(self):
        """Returns the initial population."""
        population = []

        for _ in range(self.population_size):
            solution = Solution(self.problem)
            self.problem.evaluate(solution)
            population.append(solution)

        return population
