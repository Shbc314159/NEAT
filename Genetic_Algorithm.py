import random
from Neural_Network import *
from Player import *
from Species import *
import copy
import barrier
import globalvars

class Population:
    def __init__(self, pop_size):
        self.players = []
        self.gen = 1
        self.species = []
        self.num_moves = 1000
        self.previous_best = Player()
        
        self.mass_extinction_event = False

        for _ in range(pop_size):
            player = Player()
            self.players.append(player)
            self.players[-1].brain.mutate()
    
    def update(self):
        for player in self.players:
            player.update()
    
    def speciate(self):
        for specie in self.species:
            del specie.players[:]
        
        for player in self.players:
            species_found = False
            
            for specie in self.species:
                if specie.same_species(player.brain):
                    specie.add_to_species(player)
                    species_found = True
            
            if not species_found:
                self.species.append(Species(player))
    
    def calculate_fitness(self):
        for player in self.players:
            player.fitness = player.fitness / ((len(player.brain.genome_connections) + len(player.brain.genome_neurons))/100)
    
    def sort_species(self):
        for species in self.species:
            species.sort_species()
    
        self.species.sort(key = lambda x: x.best_fitness, reverse=True)
    
    def mass_extinction(self):
        for species in range(5, len(self.species)):
            del self.species[species]
    
    def cull_species(self):
        for species in self.species:
            species.cull()
            species.fitness_sharing()
    
    def kill_stale_species(self):
        for i in range(len(self.species)-1, -1, -1):
            if self.species[i].staleness >= 15:
                del self.species[i]
        
    def get_avg_fitness_sum(self):
        return sum(s.average_fitness for s in self.species)

    def kill_bad_species(self):
        average_sum = self.get_avg_fitness_sum()
        if not average_sum:
            return None
        
        for i in range(len(self.species)-1, -1, -1):
            if self.species[i].average_fitness / average_sum * len(self.players) < 1:
                del self.species[i]
        return None

    def get_champ(self):
        highest = 0
        for player in self.players:
            if player.fitness > highest:
                champ = player
                highest = player.fitness
        
        return champ
        
    def natural_selection(self):
        previous_best = self.get_champ()
        
        self.speciate()
        self.calculate_fitness()
        self.sort_species()
        if self.mass_extinction_event:
            self.mass_extinction()
            self.mass_extinction_event = False
        self.cull_species()
        self.kill_stale_species()
        self.kill_bad_species()
        
        average_sum = self.get_avg_fitness_sum()
        if average_sum == 0:
            average_sum = 0.1
        children = []
        children.append(previous_best)
        self.previous_best = previous_best
        
        for species in self.species:
            if species.champ:
                children.append(copy.deepcopy(species.champ))
            child_count = round(species.average_fitness/average_sum * len(self.players) - 1)
            for _ in range(child_count):
                children.append(species.get_offspring())
        if len(children) < len(self.players):
            children.append(copy.deepcopy(previous_best))
        
        while len(children) < len(self.players):
            if self.species:
                children.append(self.species[0].get_offspring())
            else:
                clone = copy.deepcopy(previous_best)
                clone.brain.mutate()
                children.append(clone)
        
        self.players = children
        self.gen += 1
    
    def randomise_target(self):
        """
        randrange = int(self.current_generation/20)
        x = self.target_origin[0] - randrange + random.randint(0, randrange*2)
        z = self.target_origin[2] - randrange + random.randint(0, randrange*2)
        """
        x = random.randint(-25, 25)
        z = random.randint(-60, -10)
        
        if x < -20:
            x = -20
        elif x > 20:
            x = 20
        
        if z > -15:
            z = -15
        elif z < -55:
            z = -55
        
        globalvars.target_vertices = (
            (x + 5, -9.5, z + 5),
            (x, -9.5, z + 5),
            (x, -9.5, z),
            (x + 5, -9.5, z),
        )
        
    def randomise_barriers(self):
        globalvars.barriers = []
        globalvars.barriers_vertices = []
        
        for i in range(0):
            length = random.randint(2, 10)
            height = random.randint(2, 15)
            width = random.randint(2, 10)
            xmin = random.randint(-25, 15)
            ymin = -10
            zmin = random.randint(-60, -20)
            xmax = xmin + length
            ymax = ymin + height
            zmax = zmin + width
            newbarrier = barrier.Barrier(xmin, xmax, ymin, ymax, zmin, zmax)
            globalvars.barriers.append(newbarrier)
            globalvars.barriers_vertices.append(newbarrier.vertices)
    
def safe_randint(a, b):
    if a < b:
        return random.randint(a, b)
    elif a == b:
        return a
        
        