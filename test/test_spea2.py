from spea2 import SPEA2
# from test_problem1 import TestProblem1
from test_problem2 import TestProblem2

if __name__ == '__main__':
    # problem = TestProblem1()
    problem = TestProblem2()

    pop_size = 100
    archive_size = 100
    max_iter = 50

    archive = SPEA2(problem, pop_size, archive_size, max_iter).run(None, None)

    print(f'\nPrinting objectives for archive with {len(archive)} elements')

    for i in range(problem.num_objectives):
        print(f'\nObjective {i}:')

        for solution in archive:
            print(f'{solution.objectives[i]}')
