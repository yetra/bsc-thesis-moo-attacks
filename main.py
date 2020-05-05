import argparse
from matplotlib import pyplot as plt

from convolutional_model import ConvolutionalModel
from nsga2 import NSGA2
from problem.simple_attack import SimpleAttack
from problem.targeted_attack import TargetedAttack
from simple_model import SimpleModel
from spea2 import SPEA2


def parse_args():
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm', choices=['nsga2', 'spea2'],
                        help='the MOO algorithm to run')

    parser.add_argument('attack_type', help='the type of attack to use',
                        choices=['simple_attack', 'targeted_attack'])
    parser.add_argument('noise_size', type=float, help='the size of the noise')

    parser.add_argument('model', help='the model to attack',
                        choices=['simple_model', 'convolutional_model'])
    parser.add_argument('weights', help='path to the weights file')

    parser.add_argument('--maxiter', type=int, default=100,
                        help='the max number of algorithm iterations')
    parser.add_argument('--popsize', type=int, default=100,
                        help='the size of the population')

    return parser.parse_args()


def init_attack(args):
    """Initializes objects based on the given command-line arguments."""
    if args.model == 'simple_model':
        model = SimpleModel()
    else:
        model = ConvolutionalModel()
    model.load(args.weights)

    if args.attack_type == 'simple_attack':
        problem = SimpleAttack(model, args.noise_size)
    else:
        problem = TargetedAttack(model, args.noise_size)

    if args.algorithm == 'nsga2':
        algorithm = NSGA2(problem, args.popsize, args.maxiter)
    else:
        algorithm = SPEA2(problem, args.popsize, args.popsize, args.maxiter)

    return model, problem, algorithm
