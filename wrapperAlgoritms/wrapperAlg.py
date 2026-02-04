#VAMO a iniciar a wrapperAlg.py donde pondremos las librerias para calcular cosos de los algoritmos metahudisticos
#LIbrerias/}
from wrapperVIsual import *
import math
import matplotlib.pyplot as plt
import numpy as np
from wrapperVIsual.AM_basics import objectiveFunction

class Spehere(objectiveFunction) :
    def function(self, x):
        suma = 0
        for e in x:
            suma += e**2
        return suma
"""
La función de Ackley se utiliza ampliamente para probar algoritmos de optimización. En su 
forma bidimensional, como se muestra en el gráfico anterior, se caracteriza por una 
región exterior casi plana y un gran agujero en el centro. Esta función presenta el 
riesgo de que los algoritmos de optimización, en particular los de escalada, queden
atrapados en uno de sus numerosos mínimos locales. es como arena movediza
"""       
class Ackley(objectiveFunction):
    def function(self, x):
        n = x.shape[0]
        suma1 = 0
        suma2 = 0
        for i in range(n):
            suma1 += x[i]**2
            suma2 += math.cos(2*math.pi*x[i])
            return -20*math.exp(-0.2*math.sqrt(suma1/n)) - math.exp(suma2/n) + 20 + math.e
"""
In mathematical optimization, the Rastrigin function is a non-convex 
function used as a performance test problem for optimization algorithms.
It is a typical example of non-linear multimodal function
"""
class Rastrigin(objectiveFunction):
    def function(self, x1,x2):
       return 20 + x1**2 - 10*math.cos(2*math.pi*x1) + x2**2 - 10*math.cos(2*math.pi*x2)
"""
 La función de Levy es una función de prueba de benchmark 
 utilizada para evaluar algoritmos de optimización.
 No es un "algoritmo", sino una función objetivo compleja con 
 múltiples mínimos locales.

Características:

Multimodal: tiene muchos mínimos locales
No convexa: difícil de optimizar
Mínimo global: f(1, 1, ..., 1) = 0
"""  
   
class Levy(objectiveFunction):
    def function(self, x):
        x = np.array(x)
        n = x.shape[0]
        w = 1 + (x - 1) / 4
        
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        
        return term1 + term2 + term3
"""The Schwefel function is complex,
with many local minima. The plot shows the two-dimensional 
form of the function.
"""
class schwefel(objectiveFunction):
    def function(self, x1,x2):
        
        return 418.9829*2 - x1 * math.sin( math.sqrt( abs( x1 ))) - x2 * math.sin(math.sqrt(abs(x2)))
        """_summary_ Hay q hacer mas dimensiones
        Args:
            x1 (np.linespace): nomber,number
            x2 (np.linespace): nomber,number
        Returns:
            : _description_
        """
class RotatedHyperEllipsoid(objectiveFunction):
    def function(self, x1,x2):
         return  (2*(x1*x1)+(x2*x2))
     
"""The global minimum is inside a long, narrow, 
parabolic-shaped flat valley. To find the valley is trivial. 
To converge to the global minimum, however, is difficult.

Returns:
_type_: _description_
"""
class Rosenbrock(objectiveFunction):
    def function(self, x):
        x = np.array(x)
        suma = 0
        n = x.shape[0]
        for i in range(n-1):
            suma += 100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2
        return suma
"""The Griewank function has many widespread local minima, 
which are regularly distributed. 
The complexity is shown in the zoomed-in plots.

Returns:
_type_: _description_
"""
class Griewank(objectiveFunction):
    def function(self, x):
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        prod = 1
        for i in range(n):
            suma += (x[i]**2)/4000
            prod *= math.cos(x[i]/math.sqrt(i+1))
        return suma - prod + 1
    
    #No sure about this one
class PermOBD(objectiveFunction):
    def function(self, x):
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma_i = 0
            for j in range(n):
                suma_i += (j+1 + (x[j]**2)) / (n)
            suma += suma_i**2
        return suma

class difPotencias(objectiveFunction)  :
    def function(self, x):
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma += abs(x[i])**(i+2)
        return suma
class SumaCuadrados(objectiveFunction):
    def function(self, x):
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma += (i+1)*x[i]**2
        return suma
    

class Trid(objectiveFunction):
    def function(self, x):
        x = np.array(x)
        n = x.shape[0]
        suma1 = 0
        suma2 = 0
        for i in range(n):
            suma1 += (x[i]-1)**2
            if i < n-1:
                suma2 += x[i]*x[i+1]
        return suma1 - suma2
class Zakharov(objectiveFunction):
    def function(self, x):
        pass
class dixonPrice(objectiveFunction):
    def function(self, x):
        pass
class michalewicz(objectiveFunction):
    def function(self, x):
        pass
class permbd(objectiveFunction):
    def function(self, x):
        pass
class styblinski(objectiveFunction):
    def function(self, x):
        pass
class bochanchevski(objectiveFunction):
    def function(self, x):
        pass
        self.dim = dim

