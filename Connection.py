import random

class Connection():
    def __init__(self, input_neuron, output_neuron, innovation_number):
        self.input_neuron = input_neuron
        self.output_neuron = output_neuron
        self.innovation_number = innovation_number
        self.weight = random.random() * 4 - 2 
        self.active = True
    
    def print(self):
        print(f"Connection: {self.innovation_number} from {self.input_neuron} to {self.output_neuron}, active: {self.active}, weight: {self.weight}")
        
        