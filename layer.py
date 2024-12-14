import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Each list represents connections to a neuron from prev layer of neurons
#X = [[1, 2, 3, 2.5],
    #[2.0, 5.0, -1.0, 2.0],
    #[-1.5, 2.7, 3.3, -0.8]]
    
X, y = spiral_data(100, 3)

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class ActivationReLU:
    def forward(self, inputs):
        #ReLU is 0 if num is <= 0 but num if num > 0
        #Activation function is used to be able to fix non-linear functions
        self.output = np.maximum(0, inputs)
        
layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()
layer1.forward(X)
activation1.forward(layer1.output)