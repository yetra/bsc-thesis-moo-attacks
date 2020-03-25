from src.algorithms.nsga2 import NSGA2
from src.crossover.arithmetic_crossover import ArithmeticCrossover
from src.mutation.gaussian_mutation import GaussianMutation
from src.selection.tournament_selection import TournamentSelection
# from test_problem1 import TestProblem1
from test_problem2 import TestProblem2

if __name__ == '__main__':
    # problem = TestProblem1()
    problem = TestProblem2()

    population_size = 100
    max_iterations = 100

    crossover = ArithmeticCrossover(0.7, problem)
    mutation = GaussianMutation(0.0, 1.0, 0.03, problem)
    selection = TournamentSelection(2)

    fronts = NSGA2(problem, population_size, max_iterations,
                   crossover, mutation, selection).run()

    print(f'\nPrinting objectives for front 0 with {len(fronts[0])} elements')

    for i in range(problem.objectives_count):
        print(f'\nObjective {i}:')

        for solution in fronts[0]:
            print(f'{solution.objectives[i]}')