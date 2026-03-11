# -*- coding: utf-8 -*-
"""Algoritmo genetico para ajuste de curvas mediante representacion binaria.

El script genera una poblacion inicial, aplica torneo y cruces para minimizar
el error cuadratico medio entre una funcion parametrica y datos observados.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

#algoritmo cuadratico 

#leer datos
df = pd.read_csv('ajusteCurvas/Ajuste_de_curvas.csv')
x = np.asanyarray(df[['x']])
y = np.asanyarray(df[['y']])

plt.plot(x,y, '.b')

#variables del problema
POP_SIZE = 100 #poblacion
GENERATIONS = 20 #generacion
TOURNAMENT_PERCENT = .05 #porcentaje

lim_inf = [6,23,3,40, 9.5,15,31]
lim_sup = [10,27,5,50,12,17,36]

#funciones de ayuda

def crear_cromosoma():
    """Genera un cromosoma de 7 genes con valores enteros de 8 bits.

    Returns
    -------
    list[int]
        Secuencia de 7 enteros entre 0 y 255.
    """
    cromosoma = []
    for i in range(7):
        int_value = np.random.randint(256) #valor aleatorio entre 0 y 255
        #expandimos a cadena binaria hasta que sea la cruza
        
        cromosoma.append(int_value) #guardamos el valor, creamos la cadena de numeros
    return cromosoma

def crear_poblacion():
    """Crea la poblacion inicial de cromosomas.

    Returns
    -------
    list[list[int]]
        Poblacion con `POP_SIZE` cromosomas generados aleatoriamente.
    """
    return[crear_cromosoma() for _ in range(POP_SIZE)] #poblacion de 100, listas de 7 numeros 

def bit8_to_bin(cromosoma):
    """Convierte un cromosoma a una cadena binaria concatenada.

    Parameters
    ----------
    cromosoma : list[int]
        Secuencia de valores enteros (0-255) a transformar.

    Returns
    -------
    str
        Cadena binaria de 56 bits (7 genes * 8 bits por gen).
    """
    bit_string = ''
    for alelo in cromosoma: #por cada numero en la lista 
        bit_string += bin(alelo)[2:].zfill(8) #conviertes en forma binaria, escribe 0b a inicio, por eso es el recorte, y aparte zfill rellena de 0
    return bit_string

def bin_to_bit8(bit_string):
    """Convierte una cadena binaria concatenada a cromosoma de enteros.

    Parameters
    ----------
    bit_string : str
        Cadena de bits de longitud 56.

    Returns
    -------
    list[int]
        Lista de 7 enteros reconstruidos desde la cadena binaria.
    """
    cromosoma = []
    for i in range(7):
        cromosoma.append(int(bit_string[8*i:8*i+8], 2)) #corta en la lista cada 8
    return cromosoma

def bit8_to_float(cromosoma):
    """Mapea un cromosoma de 8 bits por gen a valores reales en rangos definidos.

    Parameters
    ----------
    cromosoma : list[int]
        Secuencia de 7 enteros entre 0 y 255.

    Returns
    -------
    list[float]
        Valores reales escalados usando los limites `lim_inf` y `lim_sup`.
    """
    v_data = []
    for i,c in enumerate(cromosoma):
        value = ((lim_sup[i]-lim_inf[i])/255)*c + lim_inf[i]
        v_data.append(value)
    return v_data

def cruce(padre1, padre2):
    """Realiza cruce de un punto entre dos padres y devuelve dos hijos.

    Parameters
    ----------
    padre1 : list[int]
        Cromosoma del primer progenitor.
    padre2 : list[int]
        Cromosoma del segundo progenitor.

    Returns
    -------
    tuple[list[int], list[int]]
        Dos cromosomas hijo resultantes del cruce.
    """
    b_padre1 = bit8_to_bin(padre1) #convertir a los padres en forma binaria
    b_padre2 = bit8_to_bin(padre2)
    #elegimos un punto de corte
    corte = np.random.randint(8*7) #en el total de la cadena agarramos un valor aleatorio
    b_hijo1 = b_padre1[:corte] + b_padre2[corte:] #desde el inicio hasta el corte del padre1, y del corte hasta el final del padre2
    b_hijo2 = b_padre2[:corte] + b_padre1[corte:]
    
    #convertir otra vez a numero
    hijo1 = bin_to_bit8(b_hijo1)
    hijo2 = bin_to_bit8(b_hijo2)
    
    return hijo1, hijo2

def funcion_de_costo(cromosoma, x, y):
    """Calcula el error cuadratico medio para un cromosoma dado.

    Parameters
    ----------
    cromosoma : list[int]
        Cromosoma a evaluar.
    x : np.ndarray
        Valores independientes de los datos observados.
    y : np.ndarray
        Valores dependientes de los datos observados.

    Returns
    -------
    float
        Error cuadratico medio entre la prediccion y los datos.
    """
    A, B, C, D, E, F, G = bit8_to_float(cromosoma) #convertir EL cromosoma a las variables
    yest = A * (B * np.sin(x/C) + D* np.cos(x/E)) + F*x - G #formula
    
    error = np.mean((y-yest)**2) #error cuadratico
    return error
    
##escribir algoritmo genetico 
def algoritmo_genetico(x, y):
    """Ejecuta el algoritmo genetico de ajuste de curvas.

    Parameters
    ----------
    x : np.ndarray
        Valores independientes de los datos.
    y : np.ndarray
        Valores dependientes de los datos.

    Returns
    -------
    list[int]
        Mejor cromosoma encontrado tras `GENERATIONS` iteraciones.
    """
    #activar modo interactivo para actualizar sin cerrar la ventana
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8)) #matriz de 1*2
    
    #crear la poblacion
    poblacion = crear_poblacion()
    
    #historial de mejor resultado
    mejores_resultados = [] #para la grafica de perdida
    
    #ciclo por cada generacion 
    for _ in range(GENERATIONS): #_ es variable anonima 
    #for anidado, por generaciones y por el torneo 
    
        #calcular fitness
        fitness = [funcion_de_costo(cromosoma, x, y) for cromosoma in poblacion]
        
        #encontrar los mejores
        mejor_idx = np.argmin(fitness)
        
        mejor_cromosoma = poblacion[mejor_idx]
        mejor_fitness = fitness[mejor_idx]
        mejores_resultados.append(mejor_fitness)
        
        #graficar funcion
        ax1.clear()
        ax1.plot(x, y, '.b')
        ax1.axis([0,100,min(y)-50, max(y)+50]) #fijando rangos de la grafica
        
        A,B,C,D,E,F,G   = bit8_to_float(mejor_cromosoma)
        yest = A * (B * np.sin(x/C) + D* np.cos(x/E)) + F*x - G #formula
        
        ax1.plot(x, yest, '-k')
        ax1.set_title("ajuste de curvas")
        
        #graficar el error
        ax2.clear()
        ax2.semilogy(range(len(mejores_resultados)), mejores_resultados, '-b')
        ax2.set_title('Historial del error')
        
        #actualizar graficos
        plt.pause(.05)
    
        #ejecutar torneo
        hijos = []
        for _ in range(POP_SIZE//2):
            permutation = random.sample(list(range(POP_SIZE)), POP_SIZE) #permuta a los individuos 
            #calcular cuantos participantes pueden participar en el torneo
            n_participantes = int(TOURNAMENT_PERCENT * POP_SIZE)
            
            #seleccionar padre1
            participantes_idx = random.sample(permutation[:POP_SIZE//2], n_participantes)
            idx_best = np.argmin([fitness[i] for i in participantes_idx])
            padre1 = poblacion[participantes_idx[idx_best]]
            
            #seleccionar padre2
            participantes_idx = random.sample(permutation[POP_SIZE//2:], n_participantes)
            idx_best = np.argmin([fitness[i] for i in participantes_idx])
            padre2 = poblacion[participantes_idx[idx_best]]
            
            #cruce
            hijo1, hijo2 = cruce(padre1,padre2)
            
            "agregar a los hijois "
            hijos.append(hijo1)
            hijos.append(hijo2)

        #remplazar a los padres
        poblacion = hijos

    #mantener la figura abierta mostrando el ultimo estado
    plt.ioff()
    plt.show()
    return mejor_cromosoma
            
    


'''
print(crear_cromosoma())
print(crear_poblacion())
'''

poblacion = crear_poblacion()
print(cruce(poblacion[3], poblacion[1]))
'''
print(poblacion[3])
bit = (bit8_to_bin(poblacion[3])) #cromosoma se convierte en lista de binarios
print(bit)
print(bin_to_bit8(bit))
print(bit8_to_float(poblacion[3]))
'''

resultado = algoritmo_genetico(x, y)
print(bit8_to_float(resultado))

