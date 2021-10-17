
from ann import NeuralNetwork

from itertools import product
import numpy as np
from random import randint, normalvariate, uniform

def examples():
    slope = uniform(-2, 2)
    inputs = np.array([slope * i + normalvariate(0, 3) for i in range(10)])
    targets = [1 if slope > 0 else 0]
    return inputs, targets

if __name__ == '__main__':
    M = NeuralNetwork.unbiased([10, 7, 4, 1])
    M.learn(examples, epochs=10**4, batch=100, verbose=lambda self: 'Energy: {:0.4f}'.format(self.energy([examples() for _ in range(100)])))
    print(' NeuralNetwork '.center(30, '='))
    print(M)
