
from itertools import product
import numpy as np
from random import randrange
from PIL import Image, ImageDraw
from ann import NeuralNetwork

def examples():
    inputs = [randrange(2) for _ in range(2)]
    targets = [inputs[0] ^ inputs[1]]
    return inputs, targets

if __name__ == '__main__':
    M = NeuralNetwork.unbiased([2, 12, 4, 1], bound_output=True)
    M.learn(examples, epochs=3*10**3, batch=100, eta=0.9, verbose=lambda self: np.array([self(i)[0] for i in product([0, 1], repeat=2)]))
    print(' NeuralNetwork '.center(30, '='))
    print(M)
