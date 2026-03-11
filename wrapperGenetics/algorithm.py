"""Entrada principal para el Algoritmo Genético."""

import random
import numpy as np
import matplotlib.pyplot as plt

from wrapperGenetics import ciudades

from functionHelper import (
    binToBit8,
	bitToBin,
	bitToFloat,
    crearPoblacion,
	generarCromosoma,
    costFunction as cost,
    operacionesHijos,
    costoRuta as costR,
    generarCromosomaCIudad,
    generarCiudadesTSP,
    crearPoblacionTSP,
    operacionesHijosTSP,
    evaluarPoblacionTSP,
)


tamanioPoblacion = 100
generaciones = 500
torneo = 0.05

def algoritmoGenetico(coords):
#Crear grafico
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,8))
    #Crear poblacion
    poblacion = [generarCromosomaCIudad() for _ in range(tamanioPoblacion)]
    #lista de resultados
    mejoresResultados = []
    for _ in range(generaciones):
        #metricaFitness
        fitness = [costR(cromosoma,coords) for cromosoma in poblacion]

        #Seleccionar los mejores
        mejoridx = np.argmin(fitness)
        mejorCromosoma = poblacion[mejoridx]
        mejorFitness = fitness[mejoridx]
        mejoresResultados.append(mejorFitness)

        #GRAFICO
        ax1.clear()
        ax1.plot(coords[:,0],coords[:,1],'.b')
        ax1.axis([0,100,min(coords[:,1])-50,max(coords[:,1])+50])
        aCoef, bCoef, cCoef, dCoef, eCoef, fCoef, gCoef = bitToFloat(mejorCromosoma)
        yEstimado = aCoef * (bCoef*np.sin(coords[:,0]/cCoef) + dCoef*np.cos(coords[:,0]/eCoef)) + fCoef * coords[:,0] - gCoef
        ax1.plot(coords[:,0],yEstimado,'-r')
        ax1.set_title(f"Mejor Cromosoma: {mejorCromosoma} - MSE: {mejorFitness:.2f}")

        #Grafico de la evolucion del error
        ax2.clear()
        ax2.semilogy(range(len(mejoresResultados)), mejoresResultados, '-o')
        ax2.set_title("Evolución del Error (MSE)")
        plt.pause(0.05)


        #TORNEO!!!!
        hijos = []
        for _ in range(tamanioPoblacion//2):
            permutacion = random.sample(range(tamanioPoblacion), tamanioPoblacion)
            numeroParticipantes = int(torneo * tamanioPoblacion)
            #seleccion de padre1
            participanteIdx =random.sample(permutacion[:tamanioPoblacion//2], numeroParticipantes)
            idx_best = np.argmin([fitness[i] for i in participanteIdx])
            padre1 = poblacion[participanteIdx[idx_best]]
            #seleccion de padre2 
            """ 
            participanteIdx =random.sample(permutacion[tamanioPoblacion//2:], numeroParticipantes)
            idx_best = np.argmin([fitness[i] for i in participanteIdx])
            padre2 = poblacion[participanteIdx[idx_best]]
            """
            #operaciones con los padres
            hijo = operacionesHijos(padre1)
            hijos.append(hijo)
        poblacion = hijos
    return mejorCromosoma    

def ejecutarAlgoritmoGenetico():    
    # Datos de ejemplo si no se proporcionan externos
    x = np.linspace(0, 100, 100)
    y = 10*np.sin(x/5) + 3*np.cos(x/7) + 0.1*x + np.random.randn(x.size)*0.5

    coords = np.column_stack((x,y))
    mejorCromosoma = algoritmoGenetico(coords)

    print("Mejor Cromosoma:", mejorCromosoma)  

def algoritmoGeneticoTSP(coords):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    poblacion = crearPoblacionTSP(tamanioPoblacion, num_ciudades=len(coords))
    mejores = []

    for _ in range(generaciones):
        fitness = evaluarPoblacionTSP(poblacion, coords)
        mejoridx = int(np.argmin(fitness))
        mejorRuta = poblacion[mejoridx]
        mejorDist = fitness[mejoridx]
        mejores.append(mejorDist)

        #Mejor disntancia obtenida
        mejorDistanciaGlobal = float("inf")
        mejorRutaGlobal = None
        mejorDistanciaIter = min(fitness)
        mejorRutaIter = poblacion[int(np.argmin(fitness))]
        if mejorDistanciaIter < mejorDistanciaGlobal:
            mejorDistanciaGlobal = mejorDistanciaIter
            mejorRutaGlobal = mejorRutaIter

        ax1.clear()
        rutaCoords = coords[mejorRuta + [mejorRuta[0]]]
        ax1.plot(rutaCoords[:, 0], rutaCoords[:, 1], '-o')
        ax1.set_title(f"Mejor dist: {mejorDist:.2f}")

        ax2.clear()
        ax2.semilogy(range(len(mejores)), mejores, '-o')
        ax2.set_title("Evolución distancia")
        plt.pause(0.05)

        hijos = []
        for _ in range(tamanioPoblacion):
            candidatos = random.sample(range(tamanioPoblacion), max(2, int(torneo * tamanioPoblacion)))
            padre = poblacion[min(candidatos, key=lambda i: fitness[i])]
            hijos.append(operacionesHijosTSP(padre))
        poblacion = hijos
        
    return mejorRuta, mejorDist, mejorRutaGlobal


def ejecutarAlgoritmoGeneticoTSP():
    num_ciudades = ciudades if isinstance(ciudades, int) else 20
    coords = generarCiudadesTSP(num_ciudades=num_ciudades, low=-100, high=100)
    mejorRuta, mejorDist, mejorDistanciaFinal = algoritmoGeneticoTSP(coords)
    print("Mejor ruta TSP:", mejorRuta)
    print("Distancia:", mejorDist)
    plt.show()


if __name__ == "__main__":
    ejecutarAlgoritmoGeneticoTSP()