

import numpy as np
from wrapperAlgoritms.wrapperAlg import *
from wrapperVIsual.AM_basics import objectiveFunction

"""Todas las funciones deben recibir un vector de n dimensiones y
regresar un solo valor, en formato pdf reporta tu código y dibuja
en 3 dimensiones (2 dimensiones de entrada y una da salida) cada una de las funciones.
"""

# Lista de clases para iterar y ejecutar pruebas fácilmente
OBJETIVE_FUNCTIONS = [
	Spehere,
	Ackley,
	Rastrigin,
	Levy,
	schwefel,
	RotatedHyperEllipsoid,
	Rosenbrock,
	Griewank,
	PermOBD,
	difPotencias,
	SumaCuadrados,
	Trid,
	Zakharov,
	dixonPrice,
	michalewicz,
	permbd,
	styblinski,
	bochanchevski,
]

vector = np.array([1,2,3,4,5])

def test_functions():
    for func_class in OBJETIVE_FUNCTIONS:
        func = func_class()
        print(f'{func_class.__name__}({vector}) = {func.function(vector)}')
        func.draw3d(name=func_class.__name__)
