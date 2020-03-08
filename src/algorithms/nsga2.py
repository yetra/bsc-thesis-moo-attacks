import math


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

    def run(self):
        """Executes the algorithm."""
        population = []
        # TODO initialize and evaluate population

        fronts = self._nondominated_sort(population)

        iteration = 0
        while iteration < self.max_iterations:
            union = self._create_union(population)
            fronts = self._nondominated_sort(union)

            next_population = []
            too_large_front = []

            for front in fronts:
                self._crowding_distance_sort(front)

                if len(next_population) + len(front) > self.population_size:
                    too_large_front = front
                    break

                next_population += front

            if len(next_population) < self.population_size:
                too_large_front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                new_last_front = []  # TODO ?

                for i in range(self.population_size - len(too_large_front)):
                    next_population += too_large_front[i]
                    new_last_front += too_large_front[i]

                fronts[-1] = new_last_front

            population = next_population

        return fronts

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

    def _crowding_distance_sort(self, front):
        """Sorts the given front by crowding distance."""
        for individual in front:
            individual.crowding_distance = 0

        front[0].crowding_distance = math.inf
        front[-1].crowding_distance = math.inf

        for i in range(self.problem.objectives_count):
            front.sort(key=lambda ind: ind.objectives[i])

            for j in range(1, len(front) - 1):
                neighbors_diff = front[j + 1].objectives[i] - front[j - 1].objectives[i]
                max_diff = self.problem.objective_maxs[i] - self.problem.objective_mins[i]
                front[j].crowding_distance += neighbors_diff / max_diff
