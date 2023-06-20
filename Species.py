from Neural_Network import *

class Species():
    def __init__(self, template_network, c1, c2, c3, compatibility_threshold):
        self.lowest_fitness = 100000
        self.last_gen_improvement = 0
        self.current_generation = 1
        self.template_network = template_network
        self.current_members = [template_network]
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.compatibility_threshold = compatibility_threshold
        self.total_fitness = 0.0

    def check_network(self, new_network):
        num_excess = 0
        num_disjoint = 0
        weight_difference_matching = 0
        genes_in_larger = 0

        if len(new_network.genome_connections) > len(self.template_network.genome_connections):
            genes_in_larger = len(new_network.genome_connections)
        else:
            genes_in_larger = len(self.template_network.genome_connections)

        newIndex = 0
        originalIndex = 0

        while newIndex < len(new_network.genome_connections) and originalIndex < len(self.template_network.genome_connections):
            newConn = new_network.genome_connections[newIndex]
            originalConn = self.template_network.genome_connections[originalIndex]

            if newConn.innovation_number == originalConn.innovation_number:
                weight_difference_matching += abs(newConn.weight - originalConn.weight)
                newIndex += 1
                originalIndex += 1
            elif newConn.innovation_number < originalConn.innovation_number:
                num_disjoint += 1
                newIndex += 1
            else:  # newConn.innovation_number > originalConn.innovation_number
                num_excess += 1
                originalIndex += 1

        num_excess += len(new_network.genome_connections) - newIndex

        difference = (self.c1 * num_excess / genes_in_larger) + (self.c2 * num_disjoint / genes_in_larger) + (self.c3 * weight_difference_matching)
        if difference < self.compatibility_threshold:
            return True
        else:
            return False

    def add_network(self, new_network):
        self.current_members.append(new_network)

    def sort_species(self):
        self.current_members.sort(key=lambda obj: obj.fitness)  

    def remove_worst(self, number_of_members):
        self.sort_species()
        prev_members = len(self.current_members)
        self.current_members = self.current_members[:-number_of_members]
        if prev_members - number_of_members != len(self.current_members):
            print("ERROR: Number of members minus number of members removed is not equal to number of members left")

    def increment_generation(self):
        self.sort_species()
        for member in self.current_members:
            if member.fitness < self.lowest_fitness:
                self.lowest_fitness = member.fitness
                self.last_gen_improvement = self.current_generation

        if (len(self.current_members) > 0):
            self.template_network = self.current_members[0]
            self.current_members.clear()
            
        self.current_generation += 1

    def kill_species(self):
        if self.last_gen_improvement < self.current_generation - 15:
            return True
        else:
            return False
        
    
inputs = [0.5, 0.5]
 
network = Neural_Network(2, 1)
for i in range(5):
    network.mutate_connection()
    network.mutate_neuron()
    network.mutate_neuron()
    
network.mutate_weights()
network2 = Neural_Network(2, 1)
for i in range(5):
    network2.mutate_connection()
    network2.mutate_neuron()
    network2.mutate_neuron()
    
network2.mutate_weights()

network3 = network.crossover(network2)

species = Species(network, 1.0, 1.0, 0.4, 3.0)

network = network3.crossover(network2)
for i in range(15):
    network.mutate_connection()
    network.mutate_neuron()
    network.mutate_neuron()