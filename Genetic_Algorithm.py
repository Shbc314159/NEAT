import random
from Neural_Network import *
from Species import *
import copy

class Genetic_Algorithm:
    def __init__(self, pop_size, mutation_probs, num_inputs, num_outputs, c1, c2, c3, compatibility_threshold):
        self.pop_size = pop_size
        self.selection_size = int(pop_size * 0.1)
        self.best_network = None
        self.current_generation = 1
        self.population = []
        self.mutation_probs = mutation_probs
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.species_list = []
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.compatibility_threshold = compatibility_threshold
        self.initialize_population()

    def initialize_population(self):
        for _ in range(self.pop_size):
            network = Neural_Network(self.num_inputs, self.num_outputs)
            network.mutate_connection()
            self.population.append(network)

    def create_next_generation(self):
        next_gen = self.speciation()
        self.population = next_gen
        self.current_generation += 1

    def speciation(self):
        num_needed = self.pop_size
        self.networks_fitness_with_complexity()
        next_gen_networks = []

        for network in self.population:
            new_species = True
            for species in self.species_list:
                if species.check_network(network):
                    species.add_network(network)
                    new_species = False
            
            if new_species:
                new_species = Species(network, self.c1, self.c2, self.c3, self.compatibility_threshold)
                new_species.add_network(network)
                self.species_list.append(new_species)

        total_fitness = 0

        for species in self.species_list:
            species.total_fitness = 0.0
            for network in species.current_members:
                network.fitness *= len(species.current_members)
                total_fitness += network.fitness
                species.total_fitness += network.fitness
            
            if len(species.current_members) == 0:
                species_avg = 0
            else:
                species_avg = species.total_fitness / len(species.current_members)
        
        all_species_avg = total_fitness / len(self.species_list)
        total_proportion = 0
    
        for species in self.species_list:
            species_proportion = 1 / (species_avg / all_species_avg)
            total_proportion += species_proportion
                
        for species in self.species_list:
            if len(species.current_members) == 0:
                num_offspring = 0
            else:
                num_offspring = int((1/(species.total_fitness/len(species.current_members)/all_species_avg)/total_proportion) * num_needed)
            num_to_remove = len(species.current_members) - num_offspring #this isn't great
        
            
            if num_to_remove == len(species.current_members) & num_offspring > 0:
                num_to_remove -= 1
                
            if num_to_remove > 0:
                species.remove_worst(num_to_remove)
                
            for _ in range(num_offspring):
                if len(species.current_members) == 1:
                    index1 = 0
                    index2 = 0
                else:
                    index1 = random.randint(0, len(species.current_members) - 1)
                    index2 = random.randint(0, len(species.current_members) - 1)
                    
                parent1 = species.current_members[index1]
                parent2 = species.current_members[index2]
                offspring = parent1.crossover(parent2)
                next_gen_networks.append(offspring)
            
            if len(species.current_members) >= 5:
                next_gen_networks.append(species.current_members[0])

            species.increment_generation()
            if species.kill_species():
                self.species_list.remove(species)

        return next_gen_networks

    def run_population(self, inputs):
        outputs = []
        for i in range(len(self.population)):
            outputs.append(network.run(inputs[i]))
        return outputs

    def get_best_network(self):
        lowest = 10000000
        for network in self.population:
            if network.fitness < lowest:
                lowest = network.fitness
                best_network = network
        return best_network

    def set_fitnesses(self, fitnesses):
        for i in range(len(self.population)):
            self.population[i].fitness = fitnesses[i]

    def networks_fitness_with_complexity(self):
        for network in self.population:
            network.fitness *= (len(network.genome_connections) * len(network.genome_neurons))