
from itertools import product
import numpy as np
from random import randrange, choice
from ann import NeuralNetwork


def examples():
    inputs = [choice([0, 1]) for _ in range(2)]
    targets = [inputs[0] ^ inputs[1]]
    return inputs, targets

def verbose(self):
    return self(np.array(list(product([0, 1], repeat=2)))).flatten()


if __name__ == '__main__':
    M = NeuralNetwork.unbiased([2, 6, 4, 1])
    M.learn(examples, epochs=10**4, batch=100, verbose=verbose)
    print(' NeuralNetwork '.center(30, '='))
    print(M)
