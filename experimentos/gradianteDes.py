import numpy as np
import matplotlib.pyplot as plt

#crear datos
n = 100 #se puede modificar
X = -1 + 2 * np.random.rand(n).reshape(1,-1) #reshape le da forma de columna
Y = -18 * X + 6 + 3*np.random.randn(n) #ecuacion lineal, randn para hacer ruido en distribucion normal, variables aleatorias suman de manera normal


#primer metodo: Gradiente descendente

beta = np.random.rand(2).reshape(1,-1) #valor inicial, es un vector   y^= B_0 + b_1x_1
epochs = 300 #epocas
eta = .3

for _ in range(epochs):
    Y_est = beta[0,0] + beta[0,1]*X #y gorrito
    beta[0,0] = beta[0,0] + (eta/n) * np.sum(Y- Y_est)
    beta[0,1] = beta[0,1] + (eta/n) *np.dot(Y-Y_est, X.T)
    beta = beta + (eta/n) * np.dot(Y-Y_est, X.T)
    
    #para sacar la linea es necesario evaluar lo que se dio con -1 y 1, tener 3 puntos y juntarlos
plt.plot([-1,1], [beta[0,0] - beta[0,1],
             beta[0,0] + beta[0,1]], '-k')

plt.plot(X,Y, '.r') #.r puntos rojos
plt.title('Gradiente descendente')


#segundo metodo : pseudoinversa

Xhat = np.concatenate((np.ones((1,n)),X),axis=0)
beta = np.dot(Y,np.linalg.pinv(Xhat))
plt.figure()
plt.plot([-1,1], [beta[0,0] - beta[0,1],
                  beta[0,0] + beta[0,1]], '-k')

plt.plot(X,Y, '.r')
plt.title('Pseudoinversa')

plt.show()