import sys
import numpy as np
import matplotlib

# Each list represents connections to a neuron from prev layer of neurons
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# Weights of each of the 4 connections for the 3 neurons
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + bias
print(output)