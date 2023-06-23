from Neural_Network import *
import math
import random

class Species():
    def __init__(self, player):
        self.best_fitness = 0.0
        self.staleness = 0
        self.current_generation = 1
        self.champ = None
        self.players = []
        self.c1 = 1.0
        self.c2 = 1.0
        self.c3 = 0.4
        self.compatibility_threshold = 5.0
        self.representative = None
        
        if player:
            self.players.append(player)
            self.best_fitness = player.fitness
            self.representative = copy.deepcopy(player.brain)
        
    @staticmethod
    def get_excess(network1, network2):
        return abs(len(network1.genome_connections) - len(network2.genome_connections))
    
    @staticmethod
    def get_disjoint(network1, network2):
        num_disjoint = 0
        if len(network1.genome_connections) > len(network2.genome_connections):
            smaller_list = network2.genome_connections
            larger_list = network1.genome_connections
        else:
            smaller_list = network1.genome_connections
            larger_list = network2.genome_connections
        
        for connection in smaller_list:
            for other_connection in larger_list:
                if connection.innovation_number == other_connection.innovation_number:
                    break
                num_disjoint += 1
                
        return num_disjoint
    
    @staticmethod
    def get_weight_difference(network1, network2):
        if not network1.genome_connections or not network2.genome_connections:
            return 0
        matching = 0
        total_difference = 0
        for connection in network1.genome_connections:
            for other_connection in network2.genome_connections:
                if connection.innovation_number == other_connection.innovation_number:
                    matching += 1
                    total_difference += abs(connection.weight - other_connection.weight)
                    break
        
        return 100 if not matching else total_difference / matching
                

    def same_species(self, new_network):
        num_excess = self.get_excess(self.representative, new_network)
        num_disjoint = self.get_disjoint(self.representative, new_network)
        weight_difference_matching = self.get_weight_difference(self.representative, new_network)
        large_genome_normalizer = max(len(new_network.genome_neurons) - 20, 1)

        compatibility = (self.c1 * num_excess / large_genome_normalizer) + (self.c2 * num_disjoint / large_genome_normalizer) + (self.c3 * weight_difference_matching)
        return self.compatibility_threshold > compatibility

    def add_to_species(self, new_member):
        self.players.append(new_member)

    def sort_species(self):
        self.players.sort(key=lambda obj: obj.fitness, reverse=True)
        if not self.players:
            self.staleness = 200
            return None

        if self.players[0].fitness > self.best_fitness:
            self.staleness = 0
            self.best_fitness = self.players[0].fitness
            self.representative = copy.deepcopy(self.players[0].brain)
        else:
            self.staleness += 1 
        return None
    
    @property
    def average_fitness(self):
        if not self.players:
            return 0.0
        return sum(player.fitness for player in self.players) / len(self.players)

    def fitness_sharing(self):
        for player in self.players:
            player.fitness /= len(self.players)
        
    def select_player(self):
        if len(self.players) == 0:
            raise RuntimeError("No players")
        fitness_sum = math.floor(sum(player.fitness for player in self.players))
        rand = 0
        if fitness_sum > 0:
            rand = random.randrange(fitness_sum)
        running_sum = 0.0
        for player in self.players:
            running_sum += player.fitness
            if running_sum > rand:
                return player
        return self.players[0]
    
    def get_offspring(self):
        if random.random() < 0.25:
            offspring = copy.deepcopy(self.select_player())
        else:
            parent1 = self.select_player()
            parent2 = self.select_player()
            
            if parent1.fitness < parent2.fitness:
                parent1, parent2 = parent2, parent1
            offspring = parent1.crossover(parent2)
        offspring.brain.mutate()
        
        return offspring

    def cull(self):
        if len(self.players) > 2:
            self.players = self.players[int(len(self.players) / 2) :]
        
        
        
"""   
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
"""