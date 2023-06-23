from OpenGL.GL import *
from OpenGL.GLU import *

import math
from PIL import Image
import globalvars
from ctypes import sizeof, c_float

class Cube():
    def __init__(self):
        self.cube_pos = [15, -9, -15]
        self.velocity = [0, 0]  
        self.acceleration = [0, 0] 
        self.friction_coefficient = 0.9
        self.gravity = -0.2
        self.force_up = 2
        self.force_horizontal = 1
        self.mass = 100
        self.direction = 45
        self.touching_ground = False
        self.colliding = False
        self.vertices = (
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, -1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, -1, 1),
            (-1, 1, 1)
            )

        self.surfaces = (
            (0,1,2,3),
            (3,2,7,6),
            (6,7,5,4),
            (4,5,1,0),
            (1,5,7,2),
            (4,0,3,6)
        )
        self.vbo_vertices = None
        self.vbo_colors = None
        
        self.current_vertices = ()
        
    def reset(self):
        self.cube_pos = [15, -9, -15]
        self.velocity = [0, 0]  
        self.acceleration = [0, 0] 
        self.touching_ground = False
    
    def move(self, movement):
        self.horizontal_movement(movement)
        self.vertical_movement(movement)
        self.set_direction(movement)
        self.velocity_acceleration(movement)
        self.collision_detections() 
#        self.get_image()
            
    
    def draw(self):
        glPushMatrix()
        glTranslatef(*self.cube_pos)
        glRotatef(self.direction, 0, 1, 0)
         
        # Create VBOs if not already created
        if self.vbo_vertices is None or self.vbo_colors is None:
            self.create_vbos()
        
        # Bind VBOs
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glColorPointer(3, GL_FLOAT, 0, None)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        
        # Draw the cube using VBOs
        glDrawArrays(GL_QUADS, 0, 24)
        
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        glPopMatrix()

    def create_vbos(self):
        vertices = [coordinate for surface in self.surfaces for vertex in surface for coordinate in self.vertices[vertex]]
        colors = [color for surface in self.surfaces for _ in surface for color in self.get_surface_color(surface)]

        vertices_data = (GLfloat * len(vertices))(*vertices)
        colors_data = (GLfloat * len(colors))(*colors)

        self.vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_data), vertices_data, GL_STATIC_DRAW)

        self.vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, sizeof(colors_data), colors_data, GL_STATIC_DRAW)

        glBindBuffer(GL_ARRAY_BUFFER, 0)

    
    def get_surface_color(self, surface):
        if surface == (6, 7, 5, 4):
            return (0, 0, 1)
        else:
            return (0, 1, 0)
    
    def horizontal_movement(self, movement):
        if movement == "w":
            self.acceleration[0] += self.force_horizontal/self.mass
        else:
            if self.velocity[0] > 0:
                self.acceleration[0] -= self.friction_coefficient
            else:
                self.acceleration[0] = 0
                self.velocity[0] = 0   
                
            if movement == "d":
                self.direction -= 5
            elif movement == "a":
                self.direction += 5
            elif movement == " ":
                self.direction += 360
    
    def vertical_movement(self, movement):
        #up/down movement calculations
        if self.over_floor() and (self.cube_pos[1] <= -9 or self.touching_ground == True):
            if movement == "u":
                self.velocity[1] = 0
                self.acceleration[1] = self.force_up
            else:
                self.acceleration[1] = 0
                
        if not self.over_floor():
            self.acceleration[1] = self.gravity
            
        if self.cube_pos[1] > -9 and self.touching_ground != True:
            self.acceleration[1] = self.gravity
    
    def set_direction(self, movement):
        #set direction and calculate angular velocity   
        if self.direction >= 360:
            self.direction -= 360
        elif self.direction <= 0:
            self.direction += 360
        
        self.direction_x = math.sin(math.radians(self.direction))
        self.direction_z = math.cos(math.radians(self.direction))
        
       
    def velocity_acceleration(self, movement): 
        #set velocities and accelerations
        self.velocity[0] += self.acceleration[0]
        self.velocity[1] += self.acceleration[1]
        
        self.cube_pos[0] += max(self.velocity[0], 0) * self.direction_x
        self.cube_pos[1] += self.velocity[1]
        self.cube_pos[2] += max(self.velocity[0], 0) * self.direction_z
        
        self.touching_ground = False
        
    def collision_detections(self):
        if self.cube_pos[2] - 1 < -60:
            self.cube_pos[2] = -59
                
        if self.over_floor() and self.cube_pos[1] < -9:
            self.cube_pos[1] = -9
            
        for barrier in globalvars.barriers:
            collisionpoints = self.collides_with(self.current_vertices, barrier.vertices)
            if collisionpoints != None:
                self.colliding = True
            else:
                self.colliding = False
            self.adjust_pos_collision(collisionpoints, barrier.vertices)
            
        self.current_vertices = self.get_current_vertices()  
        
    def over_floor(self):
        if 25 >= self.cube_pos[0] >= -25:
            if -10 >= self.cube_pos[2] >= -60:
                return True
        
        
    def point_within(self, point, xmin, xmax, ymin, ymax, zmin, zmax):
        if xmin <= point[0] <= xmax:
            if ymin <= point[1] <= ymax:
                if zmin <= point[2] <= zmax:
                    return True
            
    def collides_with(self, obj1_vertices, obj2_vertices):
        target_xpoints = [element[0] for element in obj2_vertices]
        target_ypoints = [element[1] for element in obj2_vertices]
        target_zpoints = [element[2] for element in obj2_vertices]
        
        xmax, xmin = max(target_xpoints), min(target_xpoints)
        ymax, ymin = max(target_ypoints), min(target_ypoints)
        zmax, zmin = max(target_zpoints), min(target_zpoints)
        
        pointcollided = None
        pointscollided = []
        
        for vertice in obj1_vertices:
            if self.point_within(vertice, xmin, xmax, ymin, ymax, zmin, zmax):
                pointcollided = obj1_vertices.index(vertice)
                pointscollided.append(pointcollided)
                
        return pointscollided
        
    
    def adjust_pos_collision(self, collisionpoints, vertices):
        xmin, xmax, ymin, ymax, zmin, zmax = self.vertices_minmax(vertices)
        
        if collisionpoints == [0, 1, 2, 3]:
            self.cube_pos[2] = zmax + 1
            self.acceleration[0] = 0
        elif collisionpoints == [4, 5, 6, 7]:
            self.cube_pos[2] = zmin - 1
            self.acceleration[0] = 0
        elif collisionpoints == [2, 3, 6, 7]:
            self.cube_pos[0] = xmax + 1
            self.acceleration[0] = 0
        elif collisionpoints == [0, 1, 4, 5]:
            self.cube_pos[0] = xmin - 1
            self.acceleration[0] = 0
        elif collisionpoints == [0, 3, 4, 6]:
            self.touching_ground = True
            if self.cube_pos[1] < ymax + 1: 
                self.cube_pos[1] = ymax + 1
                self.velocity[1] = 0
        elif collisionpoints == [0, 3] or \
            collisionpoints == [0, 4] or \
            collisionpoints == [3, 6] or \
            collisionpoints == [4, 6]:
            self.touching_ground = True
            if self.cube_pos[1] < ymax + 1:
                self.cube_pos[1] = ymax + 1
                
    def vertices_minmax(self, vertices):
        results = []
        for i in range(0, 3):
            column_values = [row[i] for row in vertices]
            smallest_value = min(column_values)
            largest_value = max(column_values)
            results.append(smallest_value)
            results.append(largest_value)

        return results
    
    def get_current_vertices(self):
        return (
            (self.cube_pos[0]+1, self.cube_pos[1]-1, self.cube_pos[2]-1),
            (self.cube_pos[0]+1, self.cube_pos[1]+1, self.cube_pos[2]-1),
            (self.cube_pos[0]-1, self.cube_pos[1]+1, self.cube_pos[2]-1),
            (self.cube_pos[0]-1, self.cube_pos[1]-1, self.cube_pos[2]-1),
            (self.cube_pos[0]+1, self.cube_pos[1]-1, self.cube_pos[2]+1),
            (self.cube_pos[0]+1, self.cube_pos[1]+1, self.cube_pos[2]+1),
            (self.cube_pos[0]-1, self.cube_pos[1]-1, self.cube_pos[2]+1),
            (self.cube_pos[0]-1, self.cube_pos[1]+1, self.cube_pos[2]+1)
        )
        
    def get_image(self):
        globalvars.images = globalvars.images + 1
        
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        
        try:
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)

            # create the texture object
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 80, 60, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

            # set the viewport to the size of the texture object
            glViewport(0, 0, 80, 60)

            # render the scene from the perspective of the cube
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, 80/60, 1.0, 100.0)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            camera_view = self.camera_object_pos()
            gluLookAt(self.cube_pos[0], self.cube_pos[1], self.cube_pos[2], camera_view[0], camera_view[1], camera_view[2], 0, 1, 0)

            # read the pixels from the texture object and save as an image
            pixels = glReadPixels(0, 0, 80, 60, GL_RGB, GL_UNSIGNED_BYTE)
            img = Image.frombytes("RGB", (80, 60), pixels)
            img_name = str(globalvars.images) + ".png"
            img.save("./photos/" + img_name)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.camera_reset()
            
        finally:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            glPopMatrix()
            glPopAttrib()
     
            
    def camera_reset(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (1600/1000), 0.1, 110.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0, -5, -20) 
        glRotatef(25, 2, 0, 0)
        glEnable(GL_DEPTH_TEST)   

        
    def camera_object_pos(self):
        direction_radians = math.radians(self.direction)
        
        x_offset = 10 * math.sin(direction_radians)
        z_offset = 10 * math.cos(direction_radians)
        
        object_position = [
            self.cube_pos[0] + x_offset,
            self.cube_pos[1],
            self.cube_pos[2] + z_offset
        ]
        
        return object_position
        
        
            
            