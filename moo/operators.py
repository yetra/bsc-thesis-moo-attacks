import random

import numpy as np


def reproduce(parents):
    """Reproduces the given parent population."""
    offspring = []

    while len(offspring) < len(parents):
        first_parent, second_parent = select(parents), select(parents)
        children = cross(first_parent, second_parent)

        for child in children:
            mutate(child)
            offspring.append(child)

    return offspring


def select(population, tournament_size=2):
    """Returns a solution selected from the given population."""
    best = None

    for _ in range(tournament_size):
        random_solution = random.choice(population)

        if best is None or random_solution > best:
            best = random_solution

    return best


def mutate(solution, p=0.02):
    """Mutates the given solution."""
    num_variables = solution.problem.num_variables

    perturbation = np.random.uniform(-0.02, 0.02, num_variables)
    booleans = (np.random.uniform(size=num_variables) < p).astype('float32')

    solution.variables += perturbation * booleans


def cross(parent_1, parent_2, alpha=0.7):
    """Performs arithmetic crossover on the given parent solutions."""
    problem, parent_class = parent_1.problem, parent_1.__class__

    variables = (alpha * parent_1.variables + (1 - alpha) * parent_2.variables)
    first_child = parent_class(problem, variables)

    variables = ((1 - alpha) * parent_1.variables + alpha * parent_2.variables)
    second_child = parent_class(problem, variables)

    return first_child, second_child
