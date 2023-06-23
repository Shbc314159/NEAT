from Neuron import *
from Connection import *
import globalvars
import random
import copy

class Neural_Network():
    def __init__(self, num_inputs, num_outputs):  
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.genome_neurons = []
        self.genome_connections = []
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        self.setup_layers()
    
    def setup_layers(self):
        for i in range(self.num_inputs):
            neuron = Neuron(i, None, None) 
            self.genome_neurons.append(neuron)
            self.input_neurons.append(neuron)
            self.add_to_global_list(neuron)
            
        for i in range(self.num_inputs+1, self.num_inputs+self.num_outputs+1):
            neuron = Neuron(i, None, None)
            self.genome_neurons.append(neuron)
            self.output_neurons.append(neuron)
            self.add_to_global_list(neuron)
        
        neuron = Neuron(self.num_inputs, None, None)
        self.genome_neurons.append(neuron)
        self.input_neurons.append(neuron)     
        self.add_to_global_list(neuron)
    
    def create_connection(self, input_neuron, output_neuron, weight=0.0, active=False, pass_weight=False, pass_activation=False):
        new_connection = True
        connection_in_self = False
        innovation_number = -1
        
        for connection in globalvars.connections:
            if connection.input_neuron.id == input_neuron.id and connection.output_neuron.id == output_neuron.id:
                new_connection = False
                innovation_number = connection.innovation_number
            
        if new_connection:
            innovation_number = globalvars.next_innovation_number
            globalvars.next_innovation_number += 1
        
        connection = Connection(input_neuron, output_neuron, innovation_number)
        
        if pass_weight:
            connection.weight = weight
        
        if pass_activation:
            connection.active = active
        
        for otherconnection in self.genome_connections:
            if otherconnection.innovation_number == connection.innovation_number:
                connection_in_self = True
                otherconnection.weight += connection.weight
        
        if not connection_in_self:
            self.genome_connections.append(connection)
        
        if new_connection:
            globalvars.connections.append(connection)
        
        return connection
    
    def create_neuron(self, connection):
        input_neuron = connection.input_neuron
        output_neuron = connection.output_neuron
        input_id = connection.input_neuron.id
        output_id = connection.output_neuron.id
        new_neuron = True
        neuron_in_self = False
        neuron_id = -1
        
        for neuron in globalvars.neurons:
            if neuron.input_neuron == input_id and neuron.output_neuron == output_id:
                new_neuron = False
                neuron_id = neuron.id
        
        if new_neuron:
            neuron_id = globalvars.next_id
            globalvars.next_id += 1
        
        connection.active == False
        neuron = Neuron(neuron_id, input_neuron, output_neuron)
        
        if new_neuron:
            globalvars.neurons.append(neuron)
        
        for otherneuron in self.genome_neurons:
            if otherneuron.id == neuron.id:
                neuron_in_self = True
            
        if not neuron_in_self:
            self.genome_neurons.append(neuron)
            self.hidden_neurons.append(neuron)
        
        connection1 = self.create_connection(input_neuron, neuron, 1, True, True, False)
        connection2 = self.create_connection(neuron, output_neuron, connection.weight, True, True, False)
        
        return connection1, connection2
    
    def run(self, inputs):
        neuron_map = {}
        self.reset()

        for neuron in self.genome_neurons:
            neuron_map[neuron.id] = neuron

        outputs = []
        active_connections = []
        visited = set()

        for i in range(len(inputs)):
            self.input_neurons[i].value = inputs[i]

        self.input_neurons[self.num_inputs].value = 1

        for connection in self.genome_connections:
            if connection.active == True:
                active_connections.append(connection)

        for neuron in self.output_neurons:
            outputs.append(self.get_neuron_value(neuron, active_connections, visited))
            visited.clear()

        return outputs

    def get_neuron_value(self, neuron, active_connections, visited):
        if neuron.value != None:
            return neuron.value

        if neuron.id in visited:
            return 1

        visited.add(neuron.id)

        for connection in active_connections:
            if connection.output_neuron.id == neuron.id:
                input_neuron = connection.input_neuron

                if input_neuron.value == None:
                    neuron.sum += self.get_neuron_value(connection.input_neuron, active_connections, visited) * connection.weight
                else:
                    neuron.sum += input_neuron.value * connection.weight

        neuron.activate()

        return neuron.value

    def reset(self):
        for neuron in self.genome_neurons:
            neuron.sum = 0
            neuron.value = None

    def crossover(self, other_network):
        offspring = Neural_Network(self.num_inputs, self.num_outputs)
        offspring_genes = self.match_genes(other_network)

        for connection in offspring_genes:
            offspring_connection = offspring.create_connection(connection.input_neuron, connection.output_neuron, connection.weight, connection.active, True, True)

            if not self.has_node(offspring, offspring_connection.input_neuron):
                conn_neuron = connection.input_neuron
                neuron = Neuron(conn_neuron.id, conn_neuron.input_neuron, conn_neuron.output_neuron)
                offspring.genome_neurons.append(neuron)
                id = conn_neuron.id

                if (id < self.num_inputs) or (id == self.num_inputs + self.num_outputs):
                    offspring.input_neurons.append(neuron)
                elif (id < self.num_inputs + self.num_outputs) and (id >= self.num_inputs):
                    offspring.output_neurons.append(neuron)
                else:
                    offspring.hidden_neurons.append(neuron)

            if not self.has_node(offspring, offspring_connection.output_neuron):
                conn_neuron = connection.output_neuron
                neuron = Neuron(conn_neuron.id, conn_neuron.input_neuron, conn_neuron.output_neuron)
                offspring.genome_neurons.append(neuron)
                id = conn_neuron.id

                if (id < self.num_inputs) or (id == self.num_inputs + self.num_outputs):
                    offspring.input_neurons.append(neuron)
                elif (id < self.num_inputs + self.num_outputs) and (id >= self.num_inputs):
                    offspring.output_neurons.append(neuron)
                else:
                    offspring.hidden_neurons.append(neuron)

        return offspring

    def match_genes(self, other_network):
        matching_genes = []
        self_innovation_nums = [connection.innovation_number for connection in self.genome_connections]
        other_innovation_nums = [connection.innovation_number for connection in other_network.genome_connections]
        
        all_innovation_nums = set(self_innovation_nums + other_innovation_nums)
        
        self_lined_up = []
        other_lined_up = []
        
        for innovation_num in all_innovation_nums:
            self_conn = next((connection for connection in self.genome_connections if connection.innovation_number == innovation_num), None)
            self_lined_up.append(self_conn)
            
            other_conn = next((connection for connection in other_network.genome_connections if connection.innovation_number == innovation_num), None)
            other_lined_up.append(other_conn)
        
        fitter_lined_up = self_lined_up
        worse_lined_up = other_lined_up
        
        for i in range(len(fitter_lined_up)):
            fitter_item = fitter_lined_up[i]
            other_item = worse_lined_up[i]
            
            if other_item == None:
                matching_genes.append(fitter_item)
            elif fitter_item != None and other_item != None:
                if random.random() < 0.5:
                    conn = fitter_item
                else:
                    conn = other_item
                matching_genes.append(conn)  
        
              
        return matching_genes

    def has_node(self, network, node_id):
        for neuron in network.genome_neurons:
            if neuron.id == node_id:
                return True

        return False

    def add_to_global_list(self, neuron):
        for neuron_global in globalvars.neurons:
            if neuron_global.id == neuron.id:
                return
        globalvars.neurons.append(neuron)

    def mutate_connection(self):
        input_neuron = None
        output_neuron = None

        while True:
            input_neuron = random.choice(self.genome_neurons)

            if input_neuron.id <= self.num_inputs or input_neuron.id > self.num_inputs + self.num_outputs:
                break

        while True:
            output_neuron = random.choice(self.genome_neurons)

            if output_neuron.id > self.num_inputs and output_neuron.id <= self.num_inputs + self.num_outputs:
                break

        if output_neuron.id == input_neuron.id:
            print("\nError: Input and output neurons cannot be the same\n")

        connection = self.create_connection(input_neuron, output_neuron)

    def mutate_neuron(self):
        conn = random.choice(self.genome_connections)
        self.create_neuron(conn)

    def mutate_weights(self):
        perturbation = random.random() * 4 - 2
        for connection in self.genome_connections:
            if random.random() < 0.1:
                connection.weight = random.random() * 4 - 2
            else:
                connection.weight *= perturbation

    def mutate(self):
        if 0.05 > random.random():
            self.mutate_connection()

        if 0.03 > random.random() and len(self.genome_connections) > 0:
            self.mutate_neuron()

        if 0.8 > random.random() and len(self.genome_connections) > 0:
            self.mutate_weights()
            
    def print(self):
        print("\nNeural Network:\n")
        print("Connections:")
        for connection in self.genome_connections:
            connection.print()
        print("Input neurons:")
        for neuron in self.input_neurons:
            neuron.print()
        print("Hidden neurons:")
        for neuron in self.hidden_neurons:
            neuron.print()
        print("Output neurons:")
        for neuron in self.output_neurons:
            neuron.print()
        print("Num inputs:", self.num_inputs, "Num outputs:", self.num_outputs, "Fitness:", self.fitness)
        

