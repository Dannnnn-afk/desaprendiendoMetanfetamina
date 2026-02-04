

import numpy as np
import matplotlib.pyplot as plt

class objectiveFunction:
    def __init__(self, dim, lim_inf, lim_sup):
        self.d = dim
        self.lb = lim_inf
        self.ub = lim_sup
        self.cal = False
        
    def eval(self, X):
        return np.array(list(map(self.function,list(X.T))))
    
    def function(self, x):
        pass
    
    def __calculate_draw(self,resolution=200):
        #Revisar si ya se calculo antes
        if not self.cal:
           #Revisar si se tiene la dimension adecuada
           if self.d != 2:
               raise('No podemos dibujar la funcion ya que no tiene 2 dimensiones')
               
           #crear puntos a dibujar en el dominio de la funcion 
           x = np.linspace(self.lb[0], self.ub[0], resolution)
           y = np.linspace(self.lb[1], self.ub[1], resolution)
           xx,yy = np.meshgrid(x,y)
           
           #evaluar  los puntos en la funcion
           data = (np.c_[xx.ravel(), yy.ravel()]).T
           z = self.eval(data)
           zz = z.reshape(xx.shape)
           
           self.xx = xx
           self.yy = yy
           self.zz = zz
           self.cal = True
           
           
    def draw2d(self, X=None, name=None):
        #Calcular grafico
        self.__calculate_draw()
        
        #dibujar
        fig = plt.figure(figsize=(9,9))
        plt.contourf(self.xx, self.yy, self.zz, cmap=plt.cm.coolwarm)
        plt.xlabel(r'$x$', fontsize=16)
        plt.ylabel(r'$y$', fontsize=16)
        plt.xlim(self.lb[0], self.ub[0])
        plt.ylim(self.lb[1], self.ub[1])
        
        if name is not None:
            plt.title(name, fontsize=20)
            
        if X is not None:
            plt.scatter(X[0,:], X[1,:], marker='o', c='k', s=30)
            
        return fig

    def draw3d(self, X= None, name=None):
        #calcular grafico
        self.__calculate_draw()
        
        #Dibujar
        fig = plt.figure(figsize=(9,9))
        ax= fig.add_subplot(111,projection='3d')
        ax.plot_surface(self.xx, self.yy, self.zz, 
                        cmap=plt.cm.coolwarm,
                        linewidth=0, antialiased=False, alpha=0.5)
        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$y$', fontsize=16)
        ax.set_zlabel(r'$f(x,y)$', fontsize=16)
        ax.set_xlim(self.lb[0], self.ub[0])
        ax.set_ylim(self.lb[1], self.ub[0])
        
        if name is not None:
            plt.title(name,fontsize=20)
            
        if X is not None:
            z = self.eval(X)
            ax.scatter(X[0,:], X[1,:], z, marker='o', c='k', s=30)
        return fig 
    
    def parellel_coordinates(self,X):
        
        evaluation =np.array(list(map(self.function, list(X.T))))
        norm= plt.Normalize(vmin=min(evaluation), vmax=max(evaluation))
        cmap = plt.get_cmap('coolwarm')
        
        fig, (ax1, ax2)= plt.subplots(2,1, figsize=(15,5),
                                      gridspec_kw={'height_ratios': [1,4]})
        ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax1.xaxis.set_label_position('top')
        ax1.axhline(y=0, color='k', linestyle='-')
        ax1.scatter(evaluation, np.zeros((X.shape[1])), marker='o', s=70,
                    c=cmap(norm(evaluation)))
        ax1.set_xlabel('Evaluation', fontsize=14)
        ax1.set_yticks([])
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        
        dimensions= np.arange(1, X.shape[0]+1)
        for i in range(1, X.shape[0]+1):
            ax2.axvline(x=i, color='k', linestyle='-')
        for i in range(X.shape[1]):
            ax2.plot(dimensions,X[:,i])
            
        ax2.set_xticks(dimensions)
        ax2.set_xlabel('Dimension', fontsize=14)
        ax2.set_ylabel('Value', fontsize=14)
        return fig
                
        
        
        
    
