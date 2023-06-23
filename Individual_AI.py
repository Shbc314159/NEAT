from Cube import Cube
import keyboard
import globalvars

class Individual(Cube): 
    def __init__(self, neural_network):
        super().__init__()
        self.current_move = 0
        self.moves_executed = 0
        self.score = 0
        self.fitness = 0 
        self.moves_taken = 1
        self.x_center= sum(vertex[0] for vertex in globalvars.target_vertices) / len(globalvars.target_vertices)
        self.y_center = sum(vertex[1] for vertex in globalvars.target_vertices) / len(globalvars.target_vertices)
        self.z_center = sum(vertex[2] for vertex in globalvars.target_vertices) / len(globalvars.target_vertices)
        self.neural_network = neural_network
    
    def update(self):
        self.update_target()
        outputs = self.neural_network.run([self.cube_pos[0], self.cube_pos[1], self.cube_pos[2], self.direction, self.velocity[0], 
                                        self.velocity[1], self.acceleration[0], self.acceleration[1], self.x_center, self.z_center])
        move = outputs.index(max(outputs)) 
        
        if move == 0:
            self.move("w")
        elif move == 1:
            self.move("a")
        elif move == 2:
            self.move("d") 
        elif move == 3:
            self.move("u")

            
        self.distance_from_target = ((self.x_center - self.cube_pos[0]) ** 2 + 
                            (self.y_center - self.cube_pos[1]) ** 2 + 
                            (self.z_center - self.cube_pos[2]) ** 2) ** 0.5 
        self.neural_network.fitness += self.distance_from_target
        
    def update_target(self):
        self.x_center = (globalvars.target_vertices[0][0] + globalvars.target_vertices[1][0])/2
        self.y_center = -9.5
        self.z_center = (globalvars.target_vertices[0][2] + globalvars.target_vertices[2][2])/2
 
 #deprecated function           
        
    def collide_with_target(self):
        corner1 = (self.cube_pos[0] + 1, self.cube_pos[2] - 1)
        corner2 = (self.cube_pos[0] + 1, self.cube_pos[2] + 1)
        corner3 = (self.cube_pos[0] - 1, self.cube_pos[2] + 1)
        corner4 = (self.cube_pos[0] - 1, self.cube_pos[2] - 1)
        
        targetxrange = (self.x_center + 2.5, self.x_center - 0.5)
        targetzrange = (self.z_center + 2.5, self.z_center - 2.5)
        
        if (targetxrange[0] > corner1[0] > targetxrange[1]) and (targetzrange[0] > corner1[1] > targetzrange[1]):
            return True
        elif (targetxrange[0] > corner2[0] > targetxrange[1]) and (targetzrange[0] > corner2[1] > targetzrange[1]):
            return True
        elif (targetxrange[0] > corner3[0] > targetxrange[1]) and (targetzrange[0] > corner3[1] > targetzrange[1]):
            return True
        elif (targetxrange[0] > corner4[0] > targetxrange[1]) and (targetzrange[0] > corner4[1] > targetzrange[1]):
            return True
        else:
            return False

        
        
        