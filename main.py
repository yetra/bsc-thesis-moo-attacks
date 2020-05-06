import argparse

import numpy as np
from matplotlib import pyplot as plt

import util
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


def plot_objectives(front):
    """Plots 2D objectives of solutions in the given front."""
    label_probability = [solution.objectives[0] for solution in front]
    noise_strength = [solution.objectives[1] for solution in front]

    plt.scatter(noise_strength, label_probability)
    plt.xlabel('noise strength')
    plt.ylabel('label probability')
    plt.show()


if __name__ == '__main__':
    model, problem, algorithm = init_attack(parse_args())

    x_train, y_train, _, _ = util.load_mnist(
        model.INPUT_SHAPE, model.NUM_OUTPUTS)

    for orig_image, orig_probs in zip(x_train, y_train):
        label = np.argmax(orig_probs)
        predicted_probs = model.predict(orig_image)
        if label != np.argmax(predicted_probs):
            continue

        print(f'orig_label: {label} orig_prob: {np.max(predicted_probs):.4f}')
        if isinstance(problem, TargetedAttack):
            label = (label + 1) % 10
            print(f'target_label: {label}')

        fronts = algorithm.run(orig_image, label)
        plot_objectives(fronts[0])

        for solution in fronts[0]:
            adv_probs = model.predict(orig_image + solution.variables)
            print(f'adv_label: {np.argmax(adv_probs)} {solution.objectives}')

        print()
