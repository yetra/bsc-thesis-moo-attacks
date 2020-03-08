class NSGA2:
    """An implementation of NSGA-II.

    Attributes:
        problem: the multi-objective optimization problem to solve
        population_size: the size of the population
        max_iterations: the maximum number of algorithm iterations
        crossover: the crossover operator to use
        mutation: the mutation operator to use
        selection: the selection operator to use
    """

    def __init__(self, problem, population_size, max_iterations, crossover, mutation, selection):
        """Initializes NSGA2 attributes."""
        self.problem = problem

        self.population_size = population_size
        self.max_iterations = max_iterations

        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection

    def _create_union(self, parents):
        """Returns a list containing the given parents and population_size newly generated children."""
        union = parents[:]

        child_count = 0
        while child_count < self.population_size:
            first_parent = self.selection.select_from(parents)
            second_parent = self.selection.select_from(parents)
            children = self.crossover.of(first_parent, second_parent)

            for child in children:
                self.mutation.mutate(child)  # TODO evaluate child
                union += child
                child_count += 1  # TODO check count?

        return union
