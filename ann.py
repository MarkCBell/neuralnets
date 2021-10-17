
import numpy as np
from random import choice
from itertools import product

np.set_printoptions(suppress=True, precision=4, linewidth=200)

class Activation:
    def __init__(self, function, inv_prime):
        self.function = function
        self.inv_prime = inv_prime  # The derivative of the inverse of function
    def __call__(self, value):
        return self.function(value)

sigmoid = Activation(
    lambda L: 1 / (1 + np.exp(-L)),
    lambda L: L * (1 - L)
    )

identity = Activation(
    lambda L: L,
    lambda L: 1
    )

clamp = Activation(
    lambda L: np.minimum(np.maximum(L, 0), 1),
    lambda L: np.where(np.logical_and(0 < L, L < 1), 1, 0)
    )

ReLU = Activation(  # Rectified Linear Unit
    lambda L: np.maximum(L, 0),
    lambda L: np.where(0 < L, 1, 0)
    )

arctan = Activation(
    lambda L: np.arctan(L),
    lambda L: np.power(np.cos(L), -2)  # sec^2(L)
    )

class NeuralNetwork:
    def __init__(self, weights, activations=None):
        self.weights = weights
        # We could sanity check all the shapes here.
        self.num_inputs = self.weights[0].shape[1] - 1
        self.num_outputs = self.weights[-1].shape[0]
        self.activations = activations
    
    @classmethod
    def unbiased(self, layers, activations=None):
        weights = np.array([np.array(2 * np.random.rand(h+1, w) - 1) for h, w in zip(layers, layers[1:])])
        if activations is None: activations = [sigmoid for _ in range(len(layers) - 1)]
        return NeuralNetwork(weights, activations)
    
    def __repr__(self):
        return str(self)
    def __str__(self):
        return '\n'.join(str(weight) for weight in self.weights)

    def __call__(self, inputs):
        return self.evaluate(inputs)[-1]

    @staticmethod
    def add_constant(matrix):
        return np.hstack([matrix, [[1.0] for _ in range(matrix.shape[0])]])

    def evaluate(self, inputs):
        ''' Return the capacity on each layer for the given inputs. '''

        layer = inputs
        layers = [layer]
        for weight, activation in zip(self.weights, self.activations):
            layer = activation(self.add_constant(layer).dot(weight))
            layers.append(layer)

        return layers

    def learn(self, examples, epochs=1, batch=1, learning_rate=0.9, verbose=None):
        for index in range(epochs):
            inputs, targets = zip(*[examples() for _ in range(batch)])
            inputs, targets = np.array(inputs), np.array(targets)

            layers = self.evaluate(inputs)

            layer2_delta = targets - layers[-1]
            for layer1, weight, layer2, activation in reversed(list(zip(layers, self.weights, layers[1:], self.activations))):
                delta = activation.inv_prime(layer2) * layer2_delta
                layer2_delta = delta.dot(weight.T)[:, :-1]  # Setup for next loop before we change weight.
                weight += (learning_rate / batch) * self.add_constant(layer1).T.dot(delta)

            if verbose is not None: print(index, verbose(self))

    def accuracy(self, samples):
        inputs, targets = zip(*samples)
        inputs, targets = np.array(inputs), np.array(targets)

        return np.count_nonzero(np.argmax(self(inputs), axis=1) == np.argmax(targets, axis=1)) / len(samples)

    def energy(self, samples):
        inputs, targets = zip(*samples)
        inputs, targets = np.array(inputs), np.array(targets)

        return np.linalg.norm(self(inputs) - targets, axis=1).mean()  # Max would also be reasonable.

    def gradient(self, inputs, targets):
        layers = self.evaluate(inputs)
        deltas = [targets - layers[-1]]
        
        for layer1, weight, layer2, activate in reversed(list(zip(layers, self.weights, layers[1:], self.activations))):
            layer2_delta = deltas[-1]
            delta = sigma_inv_prime(layer2) * layer2_delta if activate else layer2_delta
            weight_delta = np.outer(delta, np.hstack([layer1, [1.0]]))  # Add in the fixed +1 bias.
            deltas.append(weight_delta)
            layer1_delta = np.asarray(weight.T.dot(delta)).flatten()[:-1]  # Throw away the fixed +1 bias.
            deltas.append(layer1_delta)
        
        return np.array(deltas[::-1])
    
    def adversarial(self, inputs, targets, steps, eta, lmbda=0.05, verbose=None):
        ''' Find a small epsilon vector that can be added to inputs to try to reach targets. '''
        assert 0 < eta * lmbda < 1
        epsilon = np.random.normal(0, 0.5, len(inputs))
        for _ in range(steps):
            inputs_derivative = self.gradient(inputs + epsilon, targets)[0]
            epsilon = epsilon + eta * inputs_derivative
            epsilon *= 1 - eta * lmbda  # Shrink back towards zero vector.
            if verbose is not None: print(verbose(self, epsilon))
        
        return epsilon

