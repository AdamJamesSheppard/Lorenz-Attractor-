import random
import matplotlib.pyplot
import numpy as numpy
import pygame
from math import sin, cos


class Lorenz:
    def __init__(self):

        # initial states for the strange attractor
        self.MinXvalue, self.MaxXvalue = 30, -30
        self.minYvalue, self.MaxYvalue = -30, 30
        self.minZvalue, self.MaxZvalue = -40, 0
        # initial starting points
        self.X, self.Y, self.Z = 0.1, 0.0, 0.0
        # inital states for one lorenz attarctor at a time, this will be used for the drawing
        self.startingX, self.startingY, self.startingZ = self.X, self.Y, self.Z
        # inital states for the loops, so each lorenz attarctor will be passed through this for processing in the solve_system method
        self.initialXvalue, self.initialYvalue, self.initialZvalue = self.X, self.Y, self.Z
        self.states = None
        self.count = 0
        self.number_of_frames = 0
        self.PixelColour = (10, 100, 65)
        self.dt = 0.01
        self.tm = 10
        self.nt = int(self.tm/self.dt)
        self.t = numpy.linspace(0, self.tm, self.nt + 1)
        # Initiate states for the approximation
        self.sigma, self.beta, self.rho = 10, 8/3, 28
    # lorenz system

    def xt(self, x, y, z, t):
        return (self.sigma * (y - x))

    def yt(self, x, y, z, t):
        return (x * (self.rho - z) - y)

    def zt(self, x, y, z, t):
        return (x * y - self.beta * z)

    def Runge_Kutta_Approximation(self, xt, yt, zt, n=100000, T=10000):
        x = numpy.zeros(n + 1)  # x[k] is the solution at time t[k]
        y = numpy.zeros(n + 1)  # y[k] is the solution at time t[k]
        z = numpy.zeros(n + 1)  # z[k] is the solution at time t[k]
        t = numpy.zeros(n + 1)
        t = numpy.zeros(n + 1)
        x[0] = self.initialXvalue
        y[0] = self.initialYvalue
        z[0] = self.initialZvalue
        t[0] = 0
        dt = 0.001
        # Compute the approximate solution at equally spaced times.
        self.approximate(xt, yt, zt, n, x, y, z, t, dt)

        return x, y, z, t

    def approximate(self, xt, yt, zt, n, x, y, z, t, dt):
        for k in range(n):
            t[k+1] = t[k] + dt

            k1 = xt(x[k], y[k], z[k], t[k])
            l1 = yt(x[k], y[k], z[k], t[k])
            m1 = zt(x[k], y[k], z[k], t[k])

            k2 = xt(
                (
                    x[k] + 0.5*k1*dt),
                (
                    y[k] + 0.5*l1*dt),
                (
                    z[k] + 0.5*m1*dt),
                (
                    t[k] + dt/2)
            )
            l2 = yt(
                (
                    x[k] + 0.5*k1*dt),
                (
                    y[k] + 0.5*l1*dt),
                (
                    z[k] + 0.5*m1*dt),
                (
                    t[k] + dt/2))
            m2 = zt(
                (
                    x[k] + 0.5*k1*dt),
                (
                    y[k] + 0.5*l1*dt),
                (
                    z[k] + 0.5*m1*dt),
                (t[k] + dt/2)
            )

            k3 = xt(
                (
                    x[k] + 0.5*k2*dt),
                (
                    y[k] + 0.5*l2*dt),
                (
                    z[k] + 0.5*m2*dt),
                (t[k] + dt/2))
            l3 = yt(
                (
                    x[k] + 0.5*k2*dt),
                (
                    y[k] + 0.5*l2*dt),
                (
                    z[k] + 0.5*m2*dt),
                (t[k] + dt/2)
            )
            m3 = zt(
                (
                    x[k] + 0.5*k2*dt),
                (
                    y[k] + 0.5*l2*dt),
                (
                    z[k] + 0.5*m2*dt),
                (
                    t[k] + dt/2)
            )

            k4 = xt(
                (
                    x[k] + k3*dt),
                (
                    y[k] + l3*dt),
                (
                    z[k] + m3*dt),
                (t[k] + dt)
            )
            l4 = yt(
                (
                    x[k] + k3*dt),
                (
                    y[k] + l3*dt),
                (
                    z[k] + m3*dt),
                (
                    t[k] + dt)
            )
            m4 = zt(
                (
                    x[k] + k3*dt),
                (
                    y[k] + l3*dt),
                (
                    z[k] + m3*dt),
                (
                    t[k] + dt)
            )

            x[k+1] = x[k] + (dt*(k1 + 2*k2 + 2*k3 + k4) / 6)
            y[k+1] = y[k] + (dt*(l1 + 2*l2 + 2*l3 + l4) / 6)
            z[k+1] = z[k] + (dt*(m1 + 2*m2 + 2*m3 + m4) / 6)

    def first_round(self, xt, yt, zt, x, y, z, t):
        f0_dx = xt(x[0], y[0], z[0], t[0])
        f0_dy = yt(x[0], y[0], z[0], t[0])
        f0_dz = zt(x[0], y[0], z[0], t[0])

        f1_dx = xt(x[1], y[1], z[1], t[1])
        f1_dy = yt(x[1], y[1], z[1], t[1])
        f1_dz = zt(x[1], y[1], z[1], t[1])

        f2_dx = xt(x[2], y[2], z[2], t[2])
        f2_dy = yt(x[2], y[2], z[2], t[2])
        f2_dz = zt(x[2], y[2], z[2], t[2])

        f3_dx = xt(x[3], y[3], z[3], t[3])
        f3_dy = yt(x[3], y[3], z[3], t[3])
        f3_dz = zt(x[3], y[3], z[3], t[3])
        return f0_dx, f0_dy, f0_dz, f1_dx, f1_dy, f1_dz, f2_dx, f2_dy, f2_dz, f3_dx, f3_dy, f3_dz

    def PredictorCorrector(self, xt, yt, zt, n=100000, T=35):
        x = numpy.zeros(n + 2)  # x[k] is the solution at time t[k]
        y = numpy.zeros(n + 2)  # y[k] is the solution at time t[k]
        z = numpy.zeros(n + 2)  # z[k] is the solution at time t[k]
        t = numpy.zeros(n + 2)
        x[0] = self.initialXvalue
        y[0] = self.initialYvalue
        z[0] = self.initialZvalue
        t[0] = 0
        dt = 0.001  # 0.01

        x, y, z, t = self.Runge_Kutta_Approximation(xt, yt, zt)

        f0_dx, f0_dy, f0_dz, f1_dx, f1_dy, f1_dz, f2_dx, f2_dy, f2_dz, f3_dx, f3_dy, f3_dz = self.first_round(
            xt, yt, zt, x, y, z, t)

        self.prediction_round(xt, yt, zt, n, x, y, z, t, dt, f0_dx, f0_dy,
                              f0_dz, f1_dx, f1_dy, f1_dz, f2_dx, f2_dy, f2_dz, f3_dx, f3_dy, f3_dz)

        return x, y, z, t

    def prediction_round(self, xt, yt, zt, n, x, y, z, t, dt, f0_dx, f0_dy, f0_dz, f1_dx, f1_dy, f1_dz, f2_dx, f2_dy, f2_dz, f3_dx, f3_dy, f3_dz):
        for k in range(n-1, 0, -1):
            # Predictor: The fourth-order Adams-Bashforth technique, an explicit four-step method:
            x[k+1] = x[k] + (dt/24) * (
                55*f3_dx - 59 * f2_dx + 37*f1_dx - 9*f0_dx
            )
            y[k + 1] = y[k] + (dt / 24) * (
                55 * f3_dy - 59 * f2_dy + 37 * f1_dy - 9 * f0_dy
            )
            z[k + 1] = z[k] + (dt / 24) * (
                55 * f3_dz - 59 * f2_dz + 37 * f1_dz - 9 * f0_dz
            )

            f4_dx = xt(
                x[k+1],
                y[k+1],
                z[k+1],
                t[k+1]
            )
            f4_dy = yt(
                x[k+1],
                y[k+1],
                z[k+1],
                t[k+1]
            )
            f4_dz = zt(
                x[k + 1],
                y[k + 1],
                z[k + 1],
                t[k + 1]
            )

            # Corrector: The fourth-order Adams-Moulton technique, an implicit three-step method:
            x[k+1] = x[k] + (dt/24) * (9*xt(x[k+1],
                                            y[k+1],
                                            z[k+1],
                                            t[k+1]) + 19*f3_dx - 5*f2_dx + f1_dx
                                       )

            y[k+1] = y[k] + (dt/24) * (9*yt(x[k+1],
                                            y[k+1],
                                            z[k+1],
                                            t[k+1]) + 19*f3_dy - 5*f2_dx + f1_dy
                                       )
            z[k+1] = z[k] + (dt/24) * (9*yt(x[k+1], y[k+1],
                                            z[k+1], t[k+1]) + 19*f3_dz - 5*f2_dx + f1_dz)

    def solve_system(self):
        self.states = self.PredictorCorrector(self.xt, self.yt, self.zt)

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
        # initial conditions
        self.startingX, self.startingY, self.startingZ = self.X, self.Y, self.Z
        # X directional time stepping, this is a numerical way of stating a derivative
        # here we state the combinations of the (X,Y,Z) vector
        self.X = self.X + (self.dt * self.sigma * (self.Y - self.X))
        self.Y = self.Y + (self.dt * (self.X * (self.rho - self.Z) - self.Y))
        self.Z = self.Z + \
            (self.dt * ((self.X * self.Y) - (self.beta * self.Z)))
    '''

    def DrawToScreen(self, X, Y, MinXvalue, MaxXvalue, MinYvalue, MaxYvalue, width, height):
        newXposition = self.get_new_x_position(width)
        newYposition = self.get_new_y_position(height)
        return (newXposition), (newYposition)

    def get_new_x_position(self, width):
        newXposition = width * \
            (
                (
                    self.X - self.MinXvalue) /
                (
                    self.MaxXvalue - self.MinXvalue
                )
            )
        return newXposition

    def get_new_y_position(self, height):
        newYposition = height * \
            (
                (
                    self.Y - self.minYvalue) /
                (
                    self.MaxYvalue - self.minYvalue
                )
            )
        return newYposition

    def DrawingFunction(self, ParametricSurface):
        
        ParametricSurface = pygame.display.get_surface()
        width, height = self.get_dimensions(ParametricSurface)
        oldPosition = self.DrawToScreen(
            self.startingX,
            self.startingY,
            self.MinXvalue,
            self.MaxXvalue,
            self.minYvalue,
            self.MaxYvalue,
            width,
            height)
        newPosition = self.DrawToScreen(
            self.X,
            self.Y,
            self.MinXvalue,
            self.MaxXvalue,
            self.minYvalue,
            self.MaxYvalue,
            width, height)
        LineSegment = pygame.draw.aaline(
            ParametricSurface,
            self.PixelColour,
            oldPosition,
            newPosition,
            1)
        return LineSegment
    
    def get_dimensions(self, ParametricSurface):
        width, height = ParametricSurface.get_width(), ParametricSurface.get_height()
        return width, height
   
class LorenzApplication:
    def __init__(self):
        self.isRunning = True
        self.ParametricSurface = None
        self.fpsClock = None
        self.attractors = []
        self.size = self.width, self.height = 2048, 1080
        self.count = 0
        self.outputCount = 1

    def on_init(self):
        pygame.init()
        pygame.display.set_caption('Lorenz Attractor')
        self.displaySurface = pygame.display.set_mode((self.size))
        self.isRunning = True
        self.fpsClock = pygame.time.Clock()
        self.get_colours()

    def get_colours(self):
        attractor_colour = []
        self.random_colour(attractor_colour)
        self.append_colour_array(attractor_colour)

    def append_colour_array(self, attractor_colour):
        for i in range(0, 10):
            self.attractors.append(Lorenz())
            self.attractors[i].initialXvalue = random.uniform(-4, 4)
            self.attractors[i].PixelColour = attractor_colour[i]
            self.attractors[i].solve_system()

    def random_colour(self, attractor_colour):
        for each_attractor in range(0, 10):
            random_colours = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            attractor_colour.append(random_colours)

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self.isRunning = False

    def on_loop(self):
        for each_time in self.attractors:
            each_time.step_function()

    def on_render(self):
        for each_attractor in self.attractors:
            LineSegment = each_attractor.DrawingFunction(
                self.ParametricSurface)
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
