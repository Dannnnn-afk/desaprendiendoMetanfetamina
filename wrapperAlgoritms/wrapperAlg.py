"""Colección de funciones objetivo de prueba para metaheurísticas."""

#VAMO a iniciar a wrapperAlg.py donde pondremos las librerias para calcular cosos de los algoritmos metahudisticos
#LIbrerias/}
from wrapperVIsual import *
import math
import matplotlib.pyplot as plt
import numpy as np
from wrapperVIsual.AM_basics import objectiveFunction

class Spehere(objectiveFunction) :
    """Función Sphere.

    Suma de cuadrados de las variables. Es convexa y tiene un único mínimo global en
    el origen.
    """
    def function(self, x):
        """Evalúa la función Sphere en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        suma = 0
        for e in x:
            suma += e**2
        return suma
class Ackley(objectiveFunction):
    """Función de Ackley usada para probar algoritmos de optimización.

    En 2D se caracteriza por una región exterior casi plana y un gran agujero en el
    centro. Presenta el riesgo de que los algoritmos queden atrapados en uno de sus
    numerosos mínimos locales (como arena movediza).
    """
    def function(self, x):
        """Evalúa la función de Ackley en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        n = x.shape[0]
        suma1 = 0
        suma2 = 0
        for i in range(n):
            suma1 += x[i]**2
            suma2 += math.cos(2*math.pi*x[i])
            return -20*math.exp(-0.2*math.sqrt(suma1/n)) - math.exp(suma2/n) + 20 + math.e
class Rastrigin(objectiveFunction):
    """Función Rastrigin: prueba no convexa y multimodal.

    Es un ejemplo típico de función no lineal con muchos mínimos locales, usada para
    evaluar desempeño de algoritmos de optimización.
    """
    def function(self, x):
        """Evalúa la función de Rastrigin en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
class Levy(objectiveFunction):
    """Función Levy: benchmark con múltiples mínimos locales.

    Multimodal y no convexa. Mínimo global: $f(1, 1, ..., 1) = 0$.
    """
    def function(self, x):
        """Evalúa la función de Levy en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        w = 1 + (x - 1) / 4
        
        term1 = np.sin(np.pi * w[0])**2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        
        return term1 + term2 + term3
class schwefel(objectiveFunction):
    """Función Schwefel: compleja y con muchos mínimos locales.

    La visualización en 2D muestra un paisaje con numerosos valles locales.
    """
    def function(self, x):
        """Evalúa la función Schwefel en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
class RotatedHyperEllipsoid(objectiveFunction):
    """Función Rotated Hyper-Ellipsoid.

    Construye una suma acumulada de cuadrados que genera un valle alargado.
    """
    def function(self, x):
        """Evalúa la función Rotated Hyper-Ellipsoid en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        d = len(x)
        mat = np.tile(x, (d, 1))  # repetir xx por filas
        matlow = mat.copy()
        matlow[np.triu_indices(d, k=1)] = 0  # anular triángulo superior

        inner = np.sum(matlow ** 2, axis=1)
        outer = np.sum(inner)
        y = outer
        return y
     
class Rosenbrock(objectiveFunction):
    """Función Rosenbrock: valle estrecho con mínimo global difícil.

    El mínimo global está dentro de un valle largo y angosto con forma parabólica.
    Encontrar el valle es sencillo, pero converger al mínimo global es difícil.
    """
    def function(self, x):
        """Evalúa la función Rosenbrock en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        suma = 0
        n = x.shape[0]
        for i in range(n-1):
            suma += 100*(x[i+1]-x[i]**2)**2 + (x[i]-1)**2
        return suma
class Griewank(objectiveFunction):
    """Función Griewank: muchos mínimos locales distribuidos regularmente."""
    def function(self, x):
        """Evalúa la función Griewank en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
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
    """Función Perm OBD (variante de Perm).

    Útil para pruebas con múltiples mínimos locales.
    """
    def function(self, x):
        """Evalúa la función Perm OBD en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
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
    """Función de potencias con exponente creciente por dimensión."""
    def function(self, x):
        """Evalúa la función de potencias en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma += abs(x[i])**(i+2)
        return suma
class SumaCuadrados(objectiveFunction):
    """Suma de cuadrados ponderada por el índice de dimensión."""
    def function(self, x):
        """Evalúa la suma de cuadrados ponderada en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma += (i+1)*x[i]**2
        return suma
    

class Trid(objectiveFunction):
    """Función Trid usada para pruebas de optimización."""
    def function(self, x):
        """Evalúa la función Trid en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
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
    """Función Zakharov con términos cuadráticos y cuárticos."""
    def function(self, x):
        """Evalúa la función Zakharov en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        suma1 = 0
        suma2 = 0
        for i in range(n):
            suma1 += x[i]**2
            suma2 += 0.5*(i+1)*x[i]
        return suma1 + suma2**2 + suma2**4
    
class dixonPrice(objectiveFunction):
    """Función Dixon-Price para evaluación de algoritmos."""
    def function(self, x):
        """Evalúa la función Dixon-Price en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x1 = x[0]
        d = len(x)
        term1 = (x1 - 1) ** 2

        total = 0.0
        for ii in range(2, d + 1):
            xi = x[ii - 1]
            xold = x[ii - 2]
            new = ii * (2 * xi ** 2 - xold) ** 2
            total += new

            y = term1 + total
        return y
        
        
class michalewicz(objectiveFunction):
    """Función Michalewicz con parámetro $m=10$.

    Multimodal y con picos agudos, suele ser difícil de optimizar.
    """
    def function(self, x):
        """Evalúa la función Michalewicz en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        constant = 10
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma += math.sin(x[i]) * (math.sin((i+1)*x[i]**2/constant))**20
        return -suma

class permbd(objectiveFunction):
    """Función Perm BD con múltiples mínimos locales."""
    def function(self, x):
        """Evalúa la función Perm BD en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        b = 0.5
        suma = 0.0
        for i in range(1, n + 1):
            inner = 0.0
            for j in range(1, n + 1):
                inner += (j**i + b) * ((x[j - 1] / j) ** i - 1)
            suma += inner ** 2
        return suma
class styblinski(objectiveFunction):
    """Función Styblinski–Tang: no convexa y multimodal."""
    def function(self, x):
        """Evalúa la función Styblinski–Tang en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño $d$.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        n = x.shape[0]
        suma = 0
        for i in range(n):
            suma += x[i]**4 - 16*x[i]**2 + 5*x[i]
        return suma/2
           
class bochanchevski(objectiveFunction):
    """Función Bohachevsky (variante 1) con términos trigonométricos."""
    def function(self, x):
        """Evalúa la función Bohachevsky en el punto dado.

        Args:
            x (array-like): Vector de entrada de tamaño 2.

        Returns:
            float: Valor de la función en $x$.
        """
        x = np.array(x)
        x1 = x[0]
        x2 = x[1]
        # Bohachevsky function 1
        return x1**2 + 2 * x2**2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7
            

