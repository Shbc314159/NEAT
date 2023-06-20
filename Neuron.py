import numpy as np

class Neuron():
    def __init__(self, id, input_neuron, output_neuron):
        self.id = id
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.sum = 0
        self.value = None
         
    def activate(self):
        self.value = self.sigmoid(self.sum)
    
    def sigmoid(self, x):
        z = self.clip(x, -20, 20)
        return 1 / (1 + np.exp(-4.9*z))

    def clip(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        else:
            return x
    
    def __print__(self):
        print(f"Neuron {self.id}: {self.input_neuron} -> {self.output_neuron}, {self.sum}, {self.value}")