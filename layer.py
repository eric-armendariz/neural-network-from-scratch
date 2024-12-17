import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
class ReLUActivation:
    def forward(self, inputs):
        #ReLU is 0 if num is <= 0 but num if num > 0
        #Activation function is used to be able to fix non-linear functions
        self.output = np.maximum(0, inputs)
        
class SoftmaxActivation:
    def forward(self, inputs):
        # e**x for all batches of inputs
        # subtracted from max of its batch to normalize vals between 0 and 1
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # divide each element by sum of its batch, creates probability distribution
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
X, y = spiral_data(samples=100, classes=3)

dense1 = LayerDense(2, 3)
activation1 = ReLUActivation()
dense2 = LayerDense(3, 3)
activation2 = SoftmaxActivation()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)
