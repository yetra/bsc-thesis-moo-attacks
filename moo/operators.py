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


def mutate(solution, mu=0.0, sigma=0.3, p=0.03):  # TODO sigma related to noise_size
    """Mutates the given solution."""
    mins, maxs = solution.problem.mins, solution.problem.maxs

    perturbation = np.random.normal(mu, sigma, solution.variables.shape)
    booleans = (np.random.uniform(size=solution.variables.shape)
                < p).astype('float32')

    solution.variables += perturbation * booleans
    np.clip(solution.variables, mins, maxs, out=solution.variables)


def cross(parent_1, parent_2, alpha=0.7):
    """Performs arithmetic crossover on the given parent solutions."""
    problem, parent_class = parent_1.problem, parent_1.__class__

    variables = (alpha * parent_1.variables + (1 - alpha) * parent_2.variables)
    np.clip(variables, problem.mins, problem.maxs, out=variables)
    first_child = parent_class(problem, variables)

    variables = ((1 - alpha) * parent_1.variables + alpha * parent_2.variables)
    np.clip(variables, problem.mins, problem.maxs, out=variables)
    second_child = parent_class(problem, variables)

    return first_child, second_child
