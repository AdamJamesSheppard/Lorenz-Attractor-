import random 
import pygame 
from pygame import gfxdraw
import numpy as numpy

class Lorenz:
    def __init__(self):
        # initial states for the strange attractor 
        self.MinXvalue, self.MaxXvalue = -30, 30
        self.minYvalue, self.MaxYvalue = -30, 30
        self.minZvalue, self.MaxZvalue =  0, 50 
        # initial starting points
        self.X, self.Y, self.Z = 0.1, 0.0, 0.0
        # inital states for one lorenz attarctor at a time, this will be used for the drawing
        self.startingX, self.startingY, self.startingZ = self.X, self.Y, self.Z
        # time derivative 
        self.dt = 0.01
        self.sigma, self.rho, self.beta = 10, 23, 8/3
        self.PixelColour = (10, 100, 65)
        # inital states for the loops, so each lorenz attarctor will be passed through this for processing in the solve_system method
        self.initialXvalue, self.initialYvalue, self.initialZvalue = self.X, self.Y, self.Z
        self.states = None 
        self.count = 0
        self.number_of_frames = 0
    def xt(self, x, y, z, t):
        return (self.sigma * (y - x))

    def yt(self, x, y, z, t):
        return (self.rho*x - y - x*z)

    def zt(self, x, y, z, t):
        return (-1*(self.beta*z) + x*y)
    def Euler(self, xt, yt, zt, n = 10000,T = 35):
        x = numpy.zeros(n + 1) # x[k] is the solution at time t[k]
        y = numpy.zeros(n + 1) # y[k] is the solution at time t[k]
        z = numpy.zeros(n + 1) # z[k] is the solution at time t[k]
        t = numpy.zeros(n + 1) 
        x[0] = self.initialXvalue 
        y[0] = self.initialYvalue
        z[0] = self.initialZvalue
        t[0] = 0
        dt = T/float(n)
        for k in range(n):
            t[k + 1] = t[k] + dt
            x[k + 1] = x[k] + dt * self.xt(x[k], y[k], z[k], t[k])
            y[k + 1] = y[k] + dt * self.yt(x[k], y[k], z[k], t[k])
            z[k + 1] = z[k] + dt * self.zt(x[k], y[k], z[k], t[k])
        return x, y, z, t
    def solve_system(self):
        self.states = self.Euler(self.xt, self.yt, self.zt)
    def step_function(self):
        self.initialXvalue, self.initialYvalue, self.initialZvalue = self.X, self.Y, self.Z
        Xval = self.states[0]
        Yval = self.states[1]
        Zval = self.states[2]
        if self.count < (Xval.size - 1):
            self.X = Xval[self.count]
            self.Y = Yval[self.count]
            self.Z = Zval[self.count]
            self.count += 1

    # this function creates a smoother curve ------------------------------------------------
    '''
    def AttractorTimeStep(self):
        self.startingX, self.startingY, self.startingZ = self.X, self.Y, self.Z # initial conditions
        # X directional time stepping, this is a numerical way of stating a derivative
        # here we state the combinations of the (X,Y,Z) vector
        self.X = self.X + (self.dt * self.sigma * (self.Y - self.X))
        self.Y = self.Y + (self.dt * (self.X * (self.rho - self.Z) - self.Y))
        self.Z = self.Z + (self.dt * ((self.X * self.Y) - (self.beta * self.Z)))
    '''
    def DrawToScreen(self, X, Y, MinXvalue, MaxXvalue, MinYvalue, MaxYvalue, width, height):
        newXposition = width * ((self.X - self.MinXvalue)/ (self.MaxXvalue - self.MinXvalue))
        newYposition = height * ((self.Y - self.minYvalue)/(self.MaxYvalue - self.minYvalue))
        return (newXposition), (newYposition)
    def DrawingFunction(self, ParametricSurface):
        ParametricSurface = pygame.display.get_surface()
        width, height = ParametricSurface.get_width(), ParametricSurface.get_height()
        oldPosition = self.DrawToScreen(self.startingX, self.startingY,self.MinXvalue, self.MaxXvalue, self.minYvalue, self.MaxYvalue, width, height)
        newPosition = self.DrawToScreen(self.X, self.Y,self.MinXvalue, self.MaxXvalue, self.minYvalue, self.MaxYvalue, width, height)
        LineSegment = pygame.draw.aaline(ParametricSurface, self.PixelColour, oldPosition, newPosition, 1) 
        return LineSegment 
class LorenzApplication:
    def __init__(self):
        self.isRunning = True
        self.ParametricSurface = None
        self.fpsClock = None
        self.attractors = []
        self.size = self.width, self.height = 800, 600
        self.count = 0 
        self.outputCount = 1
    def on_init(self):
        pygame.init()
        pygame.display.set_caption('Lorenz Attractor')
        self.displaySurface = pygame.display.set_mode((self.size))
        self.isRunning = True
        self.fpsClock = pygame.time.Clock()
        attractor_colour = []
        attractor_colour.append((51, 128, 204))
        attractor_colour.append((120, 0, 100))
        attractor_colour.append((255, 191, 0))
        for i in range(0, 3):
            self.attractors.append(Lorenz())
            self.attractors[i].initialXvalue = random.uniform(-4, 4)
            self.attractors[i].PixelColour = attractor_colour[i]
            self.attractors[i].solve_system()
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.isRunning = False
    def on_loop(self):
        for each_time in self.attractors:
            each_time.step_function()
    def on_render(self):
        for each_attractor in self.attractors:
            LineSegment = each_attractor.DrawingFunction(self.ParametricSurface)
            pygame.display.update(LineSegment)
    def on_execute(self):
        if self.on_init() == False:
            self.isRunning = False 
        while self.isRunning:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
            self.fpsClock.tick()
            self.count += 1
        pygame.quit()
if __name__ == '__main__':
    t = LorenzApplication()
    t.on_execute()