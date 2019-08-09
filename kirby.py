import numpy as np
import matplotlib.pyplot as plt

#Condição Inicial

a = 1000
b = 500
n = 1000

def initial_condition_np(a,b,n):

    x = np.random.randint(-a,a, size = n)
    lim = np.around(b*np.sqrt(1-(x/a)**2))
    y = np.random.uniform(-1.0*lim,lim, size = n)#|^|repara que desse jeito ele gera uma distribuição mais densa nas pontas da elipse
    y = y.astype(int)
    #cria array com as posições das cargas
    charges = np.zeros((2,n), dtype = int)
    charges[0,:] = x
    charges[1,:] = y
    charges = np.unique(charges, axis=1) #Exclui cargas sobrepostas

    m = len(charges[0])
    print(n-m)
    """
    #Repõe Cargas Excluídas
    m = len(charges[0])
    print(n-m)
    while m < n
        x = np.random.randint(-a,a, size = n-m)
        lim = np.around(b*np.sqrt(1-(x/a)**2))
        y = np.random.uniform(-1.0*lim,lim, size = n-m)
    """        

#    plt.scatter(x,y,s=1)
#    plt.show()
    
    return charges

#Calcula o potencial da distribuição
def pot(charges):
    pot = np.zeros((len(charges[0])-1,len(charges[0])-1))
    for i in range(0,len(charges[0])-1):
        for j in range(i+1,len(charges[0])-1):
            pot[i][j] = np.log( np.sqrt( (charges[0][i] - charges[0][j])**2 + (charges[1][i] - charges[1][j])**2 ) )
            pot[j][i] = pot[i][j]
    return pot
    
#Move as Cargas
def steps(charges):
    "Andem, miseráveis!"

#plota distribuição inicial
def plote(charges):
    plt.scatter(charges[0],charges[1],s=1)
    plt.show()

#main    
charges = initial_condition_np(a,b,n)
plote(charges)
