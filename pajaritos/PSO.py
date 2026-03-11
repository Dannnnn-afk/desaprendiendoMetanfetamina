import numpy as np
import matplotlib.pyplot as plt


from wrapperVIsual.Basics import *

class PSO():
    def __init__(self, lowerLimit,upperLimit,generations=50,numParticles=20,dimension=2,w=0.6,c1=2,c2=2):
        #Set Search scope
        self.xl =lowerLimit
        self.xu = upperLimit
        #Population simulation
        self.D = dimension
        self.N = numParticles
        self.G = generations

        #PSO parameters 
        self.w = w
        self.c1 = c1
        self.c2=c2

        #Create postiion an velocity
        self.x = np.zeros((self.D,self.N)) #Current Position
        self.xb = np.zeros((self.D,self.N)) #Guardar mejores posiciones de la matris
        self.v = np.zeros((self.D,self.N)) #Velocity

        self.fx = np.zeros(self.N) #Fitness 
        self.ig=0 #Mejor global index
        self.history = np.zeros(self.G) #Guardar historia de la mejor solucion global

    def getSolution(self):
        return {'best value': self.fx[self.ig],
                'best solution': [self.x[:,self.ig]]}
    def plotHistory(self):
        plt.figure()
        plt.plot(self.history)
        plt.title('PSO History')
        plt.xlabel('Generation')
        plt.ylabel('Cost function value')

    """ F es la función objetivo a minimizar, debe ser una función que tome un vector de dimensión D y regrese un escalar"""
    def optimize(self,f,animate='no'):
        # Initialize particles aleatoriamente en el espacio de búsqueda
        self.x = self.xl + (self.xu - self.xl) * np.random.rand(self.D, self.N)
        # Inicializar velocidades en 0
        self.xb = self.x.copy() # Inicializar las mejores posiciones personales como las posiciones iniciales
        self.fx = f.eval(self.x) # Evaluar la función objetivo en las posiciones iniciales de basics
        self.ig = np.argmin(self.fx) # Encontrar el índice de la mejor posición global inicial
        #DESEMPE;O CUASI CUADRATICO de cajon, casi todos son asi, lo menciono como de examen


        Anime = None
        if animate in ['2d','3d','nd']:
            Anime = Animation(animation_type=animate, name='PSO')
            Anime.initialize_animation(f, self.x, self.ig)


        for g in range(self.G):
            if Anime is not None:
                Anime.update_animation(f, self.x, self.ig)
            for i in range(self.N): 
                # calcular velocidad
                r1 = np.random.rand(self.D)
                r2 = np.random.rand(self.D)
                self.v[:,i] = self.w * self.v[:,i] + self.c1 * r1 * (self.xb[:,i]-self.x[:,i]) + self.c2 * r2 * (self.x[:,self.ig] - self.x[:,i])
                # Actualizar posicion
                self.x[:,i] += self.v[:,i]
                # Revisar limites de busqueda
                for d in range(self.D):
                    if self.x[d,i] > self.xu[d]:
                        self.x[d,i] = self.xu[d][0]
                    if self.x[d,i] < self.xl[d]:
                        self.x[d,i] = self.xl[d][0]
                # Evaluar personal and global best
                newFitness = f.function(self.x[:,i])
                if newFitness < self.fx[i]: # Si la nueva posición es mejor que la mejor
                    self.fx[i] = newFitness
                    self.xb[:,i] = self.x[:,i].copy() # Actualizar la mejor posición personal

                if newFitness < self.fx[self.ig]: # Si la nueva posición es mejor que la mejor global
                    self.ig = i #Actualizar el indice de la mejor posición global
                

            #Guardar historia de la mejor solucion global
            self.history[g] = self.fx[self.ig]

        # Keep plot window open after optimization
        if Anime is not None:
            plt.show(block=True)






    
        




