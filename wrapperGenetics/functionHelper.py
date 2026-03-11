import numpy as np
#Variables globales
import random
ciudades = 20
def generarCromosoma():
    cromosomaLista = []
    for _ in range(7):
        intValue = np.random.randint(256)
        cromosomaLista.append(intValue)
    return cromosomaLista        

def generarCromosomaCIudad():
    return random.sample(range(ciudades), ciudades)



def crearPoblacion(tamanioPoblacion):
    return [generarCromosoma() for _ in range(tamanioPoblacion)]

def bitToBin(cromosomaLista):
    bitsString = ''
    for alelo in cromosomaLista:
        bitsString += bin(alelo)[2:].zfill(8)
    return bitsString
def bitToFloat(cromosomaLista):
    """Devuelve los 7 coeficientes como flotantes separados (A..G)."""
    return tuple(float(val) for val in cromosomaLista)
def binToBit8(bitsString):
    cromosomaLista = []
    for i in range(7):
        cromosomaLista.append(int(bitsString[i*8:(i+1)*8], 2))
    return cromosomaLista

def costFunction(cromosomaLista, xDatos, yDatos):
    aCoef, bCoef, cCoef, dCoef, eCoef, fCoef, gCoef = bitToFloat(cromosomaLista)
    yEstimado = aCoef * (bCoef*np.sin(xDatos/cCoef) + dCoef*np.cos(xDatos/eCoef)) + fCoef * xDatos - gCoef
    mseError = np.mean((yDatos - yEstimado) ** 2)
    return mseError
def costoRuta(ruta, coords):
    d = 0.0
    for i in range(len(ruta)):
        a, b = ruta[i], ruta[(i + 1) % len(ruta)]
        d += np.linalg.norm(coords[a] - coords[b])
    return d

def cruce(padreUno):
#Hacer despues
    return None

def operacionesHijos(padreUno):
    hijo = padreUno[:]  # copia
    if random.random() < 0.5:
        # mutación swap
        print("Mutación swap")
        i, j = random.sample(range(len(hijo)), 2)
        hijo[i], hijo[j] = hijo[j], hijo[i]
    else:
        # mutación por inversión de un segmento
        print("Mutación inversión")
        i, j = sorted(random.sample(range(len(hijo)), 2))
        hijo[i:j] = reversed(hijo[i:j])
    return hijo


# === Funciones adicionales para TSP (20 ciudades en [-100, 100]) ===

def generarCiudadesTSP(num_ciudades=20, low=-100.0, high=100.0, rng=None):
    """Genera coordenadas aleatorias para las ciudades en el rango [low, high]."""
    rng = rng or np.random.default_rng()
    return rng.uniform(low, high, size=(num_ciudades, 2))


def crearPoblacionTSP(tamanioPoblacion, num_ciudades=20):
    """Crea una población de rutas (permutaciones)."""
    return [random.sample(range(num_ciudades), num_ciudades) for _ in range(tamanioPoblacion)]


def operacionesHijosTSP(padreUno):
    """Solo mutación (swap o inversión) sobre una ruta TSP."""
    hijo = padreUno[:]
    if random.random() < 0.5:
        i, j = random.sample(range(len(hijo)), 2)
        hijo[i], hijo[j] = hijo[j], hijo[i]
    else:
        i, j = sorted(random.sample(range(len(hijo)), 2))
        hijo[i:j] = reversed(hijo[i:j])
    return hijo


def evaluarPoblacionTSP(poblacion, coords):
    """Calcula la distancia total de cada ruta en la población."""
    return [costoRuta(ruta, coords) for ruta in poblacion]

"""def reemplazarSegmentoLista(cromosomaLista, startDest, endDest, startSrc, endSrc):
    Reemplaza cromosoma[start_dest:end_dest) por cromosoma[start_src:end_src).
    longitud = len(cromosomaLista)
    startDest, endDest = max(0, startDest), min(longitud, endDest)
    startSrc, endSrc = max(0, startSrc), min(longitud, endSrc)
    nuevoCromosoma = list(cromosomaLista)
    nuevoCromosoma[startDest:endDest] = nuevoCromosoma[startSrc:endSrc]
    return nuevoCromosoma

def invertirSegmentoLista(cromosomaLista, startIndex, endIndex):
    Invierte en sitio el segmento cromosoma[start:end).
    longitud = len(cromosomaLista)
    startIndex, endIndex = max(0, startIndex), min(longitud, endIndex)
    nuevoCromosoma = list(cromosomaLista)
    nuevoCromosoma[startIndex:endIndex] = reversed(nuevoCromosoma[startIndex:endIndex])
    return nuevoCromosoma
        



"""