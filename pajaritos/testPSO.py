import numpy as np
from PSO import *
from wrapperAlgoritms.wrapperAlg import *
from wrapperVIsual.Basics import *




"""
Paso 2:  Adicional al código desarrollado con el profesor, crea un codígo adicional que haga lo siguiente:
- Elabora un script puedas correr multiples veces el algoritmo y te entregué el tiempo medio y su desviación estandar, la solución óptima y su desviación estándar.
- Muestra tu código y resultados de 30 ejecuciones sobre las funciónes Ackley, Rastringin y Sphere, en 3 dimensiones de entreda.
- Agrega al reporte tu código y resultados.

Paso 3:
- Utilizando la función Ackley, experimenta con diferentes valores de w, c1, y c2. Visualiza las animaciones.
- Escribe en tu reporte lo que aprendiste de mariar estos valores (al menos media cuartilla)."""




#Espacios de busquedas 
#dimensiones, limites y funciones objetivo
dim = 10
xl= -5 * np.ones((dim,1))
xu = 5 * np.ones((dim,1))
f = Ackley(dim,xl,xu)

algoritmo = PSO(xl,xu,generations=100,numParticles=50,dimension=dim,w=0.6,c1=2,c2=2)
algoritmo.optimize(f,animate='nd')
print(algoritmo.getSolution())
algoritmo.plotHistory()
