import math

from moo import operators
from moo.solution.nsga2_solution import NSGA2Solution


class NSGA2:
    """An implementation of NSGA-II.

    Attributes:
        problem: the multi-objective optimization problem to solve
        pop_size: the size of the population
        max_iterations: the maximum number of algorithm iterations
    """

    def __init__(self, problem, pop_size, max_iterations):
        """Initializes NSGA2 attributes."""
        self.problem = problem
        self.pop_size = pop_size
        self.max_iterations = max_iterations

    def run(self, orig_image, label):
        """Executes the algorithm."""
        population = self.initialize()

        self.problem.evaluate(population, orig_image, label)
        self.fast_non_dominated_sort(population)

        iteration = 0
        while iteration < self.max_iterations:
            # print(f'i={iteration}')

            offspring = operators.reproduce(population)
            self.problem.evaluate(offspring, orig_image, label)

            fronts = self.fast_non_dominated_sort(population + offspring)
            population = self.build_population(fronts)

            iteration += 1

        first_front = self.fast_non_dominated_sort(population)[0]

        return first_front

    def initialize(self):
        """Returns the initial population."""
        return [NSGA2Solution(self.problem) for _ in range(self.pop_size)]

    def build_population(self, fronts):
        """Builds a population from the given fronts."""
        population = []

        for front in fronts:
            if len(population) + len(front) > self.pop_size:
                front.sort(reverse=True)
                fill_count = self.pop_size - len(front)
                population += front[:fill_count]
                break

            population += front

        return population

    def fast_non_dominated_sort(self, population):
        """
        Returns the fronts obtained by performing a non-dominated sort of
        the given population.

        :param population: the population to sort
        :return: the fronts
        """
        current_front = []

        for solution in population:
            solution.dominates_list = []
            solution.dominated_by = 0

            for candidate in population:
                if solution.dominates(candidate):
                    solution.dominates_list.append(candidate)
                elif candidate.dominates(solution):
                    solution.dominated_by += 1

            if solution.dominated_by == 0:
                current_front.append(solution)
                solution.rank = 0

        fronts = []
        front_index = 0

        while current_front:
            self.crowding_distance_assignment(current_front)
            fronts.append(current_front)
            next_front = []

            for solution in current_front:
                for dominated in solution.dominates_list:
                    dominated.dominated_by -= 1

                    if dominated.dominated_by == 0:
                        next_front.append(dominated)
                        dominated.rank = front_index + 1

            current_front = next_front
            front_index += 1

        return fronts

    def crowding_distance_assignment(self, front):
        """
        Assigns crowding distance values to all solutions in the given front.

        :param front: the front containing solutions
        """
        for solution in front:
            solution.crowding_distance = 0

        front[0].crowding_distance = math.inf
        front[-1].crowding_distance = math.inf

        for i in range(self.problem.num_objectives):
            front.sort(key=lambda ind: ind.objectives[i])

            for j in range(1, len(front) - 1):
                neighbors_diff = (front[j + 1].objectives[i]
                                  - front[j - 1].objectives[i])

                max_diff = self.problem.o_maxs[i] - self.problem.o_mins[i]

                front[j].crowding_distance += neighbors_diff / max_diff
