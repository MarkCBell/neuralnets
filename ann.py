
import numpy as np
from random import choice
from itertools import product

np.set_printoptions(suppress=True, precision=4, linewidth=200)

def sigma(L): return 1 / (1 + np.exp(-L))  # Sigmoid
def sigma_inv_prime(L): return L * (1 - L)
# def sigma(L): return np.clip(L, 0, 1)
# def sigma_inv_prime(L): return np.where(np.logical_and(0 < L, L < 1), 1, 0)
# def sigma(L): return np.maximum(L, 0)
# def sigma_inv_prime(L): return np.where(0 < L, 1, 0)

class NeuralNetwork:
    def __init__(self, weights, activations=None):
        self.weights = weights
        # We could sanity check all the shapes here.
        self.num_inputs = self.weights[0].shape[1] - 1
        self.num_outputs = self.weights[-1].shape[0]
        self.activations = activations if activations is not None else [True] * len(self.weights)
    
    @classmethod
    def unbiased(self, shapes, activations=None, bound_output=False):
        weights = np.array([np.matrix(2 * np.random.rand(h, w+1) - 1) for h, w in zip(shapes[1:], shapes)])
        if activations is None:
            activations = [True] * (len(weights) - 1) + [bound_output]
        return NeuralNetwork(weights, activations)
    
    def __repr__(self):
        return str(self)
    def __str__(self):
        return '\n'.join(str(weight) for weight in self.weights)
    
    def __call__(self, inputs):
        return self.evaluate(inputs)[-1]
    
    def accuracy(self, samples):
        return sum(1 if np.argmax(targets) == np.argmax(self(inputs)) else 0 for inputs, targets in samples) / len(samples)
    
    def energy(self, samples):
        return sum(np.linalg.norm(self(inputs) - targets) for inputs, targets in samples) / len(samples)
    
    def evaluate(self, inputs):
        ''' Return the capacity on each layer for the given inputs. '''
        inputs = np.array(inputs)
        
        assert len(inputs) == self.num_inputs
        
        layer = inputs
        layers = [layer]
        for weight, activate in zip(self.weights, self.activations):
            layer = np.hstack([layer, [1.0]])
            layer = np.asarray(weight.dot(layer)).flatten()
            if activate: layer = sigma(layer)
            layers.append(layer)
        
        return layers
    
    def gradient(self, inputs, targets):
        inputs = np.array(inputs)  # Not needed.
        targets = np.array(targets)
        
        assert len(inputs) == self.num_inputs
        assert len(targets) == self.num_outputs
        
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
    
    def learn(self, examples, epochs=1, batch=1, eta=0.9, verbose=None):
        ''' Modify self to match the provided examples. '''
        for i in range(epochs):
            self.weights += (eta / batch) * sum(self.gradient(*examples())[1::2] for _ in range(batch))
            if verbose is not None: print(i, verbose(self))
    
    def adversarial(self, inputs, targets, steps, eta, lmbda=0.05, verbose=None):
        ''' Find a small epsilon vector that can be added to inputs to try to reach targets. '''
        epsilon = np.random.normal(0, 0.5, len(inputs))
        for _ in range(steps):
            inputs_derivative = self.gradient(inputs + epsilon, targets)[0]
            epsilon = epsilon + eta * inputs_derivative
            epsilon = epsilon - eta * lmbda * epsilon  # Pull back towards zero vector.
            if verbose is not None: print(verbose(self, epsilon))
        
        return epsilon

