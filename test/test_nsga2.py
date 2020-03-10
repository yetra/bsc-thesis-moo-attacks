from src.algorithms.nsga2 import NSGA2
from src.crossover.arithmetic_crossover import ArithmeticCrossover
from src.mutation.gaussian_mutation import GaussianMutation
from src.selection.tournament_selection import TournamentSelection
from test.test_problem1 import TestProblem1

if __name__ == '__main__':
    problem = TestProblem1()
    # problem = TestProblem2()

    population_size = 100
    max_iterations = 1000

    crossover = ArithmeticCrossover(0.5)
    mutation = GaussianMutation(0.0, 1.0, 0.03)
    selection = TournamentSelection(2)

    NSGA2(problem, population_size, max_iterations, crossover, mutation, selection).run()
