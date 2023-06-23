import pygame
from pygame.locals import *
from Genetic_Algorithm import Population

from OpenGL.GL import *
from OpenGL.GLU import *

from Walls import Walls
from graph import Graph
import globalvars 

import gc
import cProfile

def main(): 
    
    pygame.init()
    display = (1600,1000)
    pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, 10)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL|RESIZABLE)

    gluPerspective(45, (display[0]/display[1]), 0.1, 110.0)
    
    glTranslatef(0, -5, -20)
    glRotatef(25, 2, 0, 0)
    
    glEnable(GL_DEPTH_TEST)
    
    walls = Walls()
    pop = Population(150)    
    graph = Graph()
    
    while pop.gen < 1000:
        print("Generation:", pop.gen)
        
        for i in range(pop.num_moves):
            draw_stuff(walls, pop)
            pop.update()
            
        pop.randomise_target()
        
        for individual in pop.players:
            individual.set_fitness()
        
        update_graph(pop, graph)
        pop.natural_selection()
        
def draw_stuff(walls, pop):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glClearColor(0, 0, 0, 1)

    walls.Draw_Back_Wall()
    walls.Draw_Floor()
    walls.Draw_Target()
    
    for barrier in globalvars.barriers:
        barrier.draw()
    
    for individual in pop.players:
        if individual == pop.previous_best or pop.gen == 1:     
            individual.draw()
    
    pygame.display.flip()

def update_graph(genetic_algorithm, graph):
    total = 0
    for individual in genetic_algorithm.players:
        total += individual.fitness
    
    x = genetic_algorithm.gen
    y = total / len(genetic_algorithm.players)
    print(y)
    graph.update(x, y)
    
    
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    profiler.dump_stats('profile_results.prof')
   
        