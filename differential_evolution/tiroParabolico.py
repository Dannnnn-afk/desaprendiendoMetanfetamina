# Tiro parabolico
#IMportar bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
#from DE import *
from pajaritos.PSO import *



#Funciones para simulacion
def tiroParabolico(angle,velocity,g=9.81,dt=0.01):
    #Convertir a radianes
    angleRad= np.radians(angle)



    #Condiciones iniciales
    vx = velocity *np.cos(angleRad)
    vy = velocity * np.sin(angleRad)

    x,y = 0.0,0.0
    trajectory = []

    #simular tiro 
    while y >= 0.0:
        trajectory.append([x,y])
        #acutalizar posicion 
        x += vx * dt
        y += vy * dt 
        vy -= g*dt

    return np.array(trajectory)

def plot_trajectory(trajectory):
    plt.figure()
    plt.plot(trajectory[:,0], trajectory[:,1],'-k')
    plt.xlabel('distance')
    plt.ylabel('altura')
    plt.axis('equal')
    plt.title('tiro parabolico')
    plt.grid(True)
    plt.show()

def createTarjet():
    return 5 + 20 *np.random.rand(1,2)

def plotTarjet(target):
    plt.scatter(target[0,0], target[0,1], s=100, c='r')

def mindistance(trajectory,target):
    return np.min(cdist(trajectory,target))

#codigo test
""" 
angle=45
velocity = 10
trajectory = tiroParabolico(angle,velocity)
plot_trajectory(trajectory)
target = createTarjet()
plotTarjet(target)
min_dist = mindistance(trajectory,target)
print(f"Distancia mínima: {min_dist}")

"""


#funcion objetivo
class matadrones(Objective_function):
    def function(self, x):
        angle = x[0]
        vel = x[1]
        trajectory = tiroParabolico(angle, vel)
        return mindistance(trajectory,target)**2



#optimizar 
target = createTarjet()
#mover cosos de aqui cuando de cosas raras
xl = np.array([[0],[0]])
xu= np.array([[90],[50]])#instanciar algoritmo

f= matadrones(2,xl,xu)
algpso = PSO(xl,xu,generations=20,numParticles=20,dimension=2,w=0.6,c1=2,c2=2)

optimizar = algpso.optimize(f,animate='2d')
print(algpso.getSolution())


trajectory_opt = tiroParabolico(algpso.getSolution()['best solution'][0][0], algpso.getSolution()['best solution'][0][1])
plot_trajectory(trajectory_opt)
plotTarjet(target)
