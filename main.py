import argparse
import csv

import numpy as np
from PIL import Image
from keras.datasets import mnist
from matplotlib import pyplot as plt

import util
from models.convolutional_model import ConvolutionalModel
from models.simple_model import SimpleModel
from moo.nsga2 import NSGA2
from moo.problem.improved_targeted_attack import ImprovedTargetedAttack
from moo.problem.simple_attack import SimpleAttack
from moo.problem.targeted_attack import TargetedAttack
from moo.spea2 import SPEA2
from util import load_mnist


def parse_args():
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('algorithm', choices=['nsga2', 'spea2'],
                        help='the MOO algorithm to run')

    parser.add_argument('attack_type', help='the type of attack to use',
                        choices=['simple_attack', 'targeted_attack',
                                 'improved_attack'])
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
    elif args.attack_type == 'targeted_attack':
        problem = TargetedAttack(model, args.noise_size)
    else:
        problem = ImprovedTargetedAttack(model, args.noise_size)

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


def save_image(array, filename):
    """Saves the given array as an image."""
    array = (array * 255).reshape(28, 28).astype('uint8')
    Image.fromarray(array, 'L').save(filename)


if __name__ == '__main__':
    header = ['sample_idx', 'orig_label', 'orig_prob', 'adv_label', 'adv_prob',
              'adv_orig_prob', 'obj_0', 'obj_1', 'adv_probs', 'succ']
    target_label = 3

    np.set_printoptions(formatter={'float': lambda x: f'{x:.4f}'})

    model, problem, algorithm = init_attack(parse_args())

    x_test = load_mnist(model.INPUT_SHAPE, model.NUM_OUTPUTS)[2]
    y_test = mnist.load_data()[1][1]
    x_rand, y_rand = util.sample_choice(x_test, y_test, range(10), 3, seed=43)

    csv_file = open('data.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(header)

    print(','.join(header))

    for sample_idx, (orig_image, label) in enumerate(zip(x_rand, y_rand)):
        probs = model.predict(orig_image)
        orig_prob = probs[label]

        if label != np.argmax(probs):
            continue
        if not isinstance(problem, SimpleAttack):
            if label == target_label:
                continue
            label = target_label

        save_image(orig_image, filename=f'sample{sample_idx}_orig.png')

        results = algorithm.run(orig_image, label)
        # plot_objectives(results)

        for adv_idx, solution in enumerate(results):
            adv_image = orig_image + solution.variables
            adv_probs = model.predict(adv_image)
            adv_label = np.argmax(adv_probs)

            save_image(adv_image, filename=f'sample{sample_idx}_{adv_idx}.png')

            line = [sample_idx, label, orig_prob, adv_label,
                    adv_probs[adv_label], adv_probs[label],
                    solution.objectives[0], solution.objectives[1],
                    adv_probs, adv_label == label]
            csv_writer.writerow(line)

            print(','.join(str(e) for e in line))

    csv_file.close()
