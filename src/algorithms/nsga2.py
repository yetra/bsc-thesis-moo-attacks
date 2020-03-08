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

    def _nondominated_sort(self, population):
        """Returns the fronts obtained by performing a non-dominated sort of the given population."""
        current_front = []

        for individual in population:
            individual.dominates_list = []

            for candidate in population:
                if individual.dominates(candidate):
                    individual.dominates_list += candidate
                elif candidate.dominates(individual):
                    individual.dominated_by += 1

            if individual.dominated_by == 0:
                current_front += individual
                individual.rank = 0

        fronts = []
        front_index = 0

        while current_front:
            fronts.append(current_front)
            next_front = []

            for individual in current_front:
                for dominated_individual in individual.dominates_list:
                    dominated_individual.dominated_by -= 1

                    if dominated_individual.dominated_by == 0:
                        next_front += dominated_individual
                        dominated_individual.rank = front_index + 1

            front_index += 1
            current_front = next_front

        return fronts
