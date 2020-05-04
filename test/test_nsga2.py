from moo.nsga2 import NSGA2
# from test_problem1 import TestProblem1
from test_problem2 import TestProblem2

if __name__ == '__main__':
    # problem = TestProblem1()
    problem = TestProblem2()

    pop_size = 100
    max_iter = 100

    fronts = NSGA2(problem, pop_size, max_iter).run(None, None)

    print(f'\nPrinting objectives for front 0 with {len(fronts[0])} elements')

    for i in range(problem.objectives_count):
        print(f'\nObjective {i}:')

        for solution in fronts[0]:
            print(f'{solution.objectives[i]}')
