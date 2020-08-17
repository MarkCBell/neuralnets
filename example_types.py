
from ann import NeuralNetwork

from itertools import product
import numpy as np
from random import randrange

LENGTH = 30

def examples(mode=None, lower=0, upper=+0):
    if mode is None: mode = randrange(4)
    inputs = np.ones(LENGTH) * np.random.uniform(lower, upper) + np.random.normal(0, 0.5, LENGTH)
    if mode == 0:
        targets = [1, 0, 0, 0]
    elif mode == 1:
        start = randrange(LENGTH-6)
        inputs[start+0:start+6] += +1
        targets = [0, 1, 0, 0]
    elif mode == 2:
        start = randrange(LENGTH-6)
        inputs[start+0:start+6:2] += +1
        inputs[start+1:start+6:2] += -1
        targets = [0, 0, 1, 0]
    else:  # mode == 3:
        start = randrange(LENGTH-6)
        inputs[start+0:start+3] += +1
        inputs[start+3:start+6] += -1
        targets = [0, 0, 0, 1]
    return inputs, targets

if __name__ == '__main__':
    M = NeuralNetwork.unbiased([LENGTH, 12, 4], bound_output=True)
    M.learn(examples, epochs=10**4, batch=100, eta=0.9,
        verbose=lambda self: 'Accuracy: {1:0.1f}% Energy: {0:0.4f} (lower == better ML)'.format(
            self.energy([examples() for _ in range(100)]),
            100 * self.accuracy([examples() for _ in range(100)])
            )
        )
    print(' NeuralNetwork '.center(30, '='))
    print(M)

    for i in range(4):
        print('Mode {}: Accuracy {}'.format(i, sum(M(examples(i)[0]) for _ in range(100)) / 100.0))

    print(' Adversary example '.center(30, '='))
    while True:
        inputs, targets = examples(2)
        if np.argmax(M(inputs)) == 2: break

    fake_targets = [0, 0, 0, 1]
    epsilon = M.adversarial(inputs, fake_targets, 10000, eta=0.4, lmbda=0.5)  # Find a nudge that gets us close to fake_target.
    print('Inputs: ', inputs)
    print('Target: ', targets)
    print('Output: ', M(inputs))
    print('Epsilon: ', epsilon)
    print('Nudged output: ', M(inputs + epsilon))
    print('Size of epsilon: {} {}'.format(np.linalg.norm(epsilon), np.linalg.norm(epsilon, np.inf)))
    
    for i in range(0, 100, 5):
        print(i, M(inputs + (i / 100.0) * epsilon))
    

