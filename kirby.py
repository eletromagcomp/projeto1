import numpy as np
import matplotlib.pyplot as plt
import math as m

#Condição Inicial

a = 1000
b = 500
n = 50

def initial_condition_np(a,b,n):

    x = np.random.randint(-a,a, size = n)
    lim = np.around(b*np.sqrt(1-(x/a)**2))
    y = np.random.uniform(-1.0*lim,lim, size = n)#|^|repara que desse jeito ele gera uma distribuição mais densa nas pontas da elipse
    y = y.astype(int)
    #cria array com as posições das cargas
    charges = np.zeros((n,2), dtype = int)
    charges[:,0] = x
    charges[:,1] = y
    charges = np.unique(charges, axis=0) #Exclui cargas sobrepostas

    m = len(charges)
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
     
    return charges

#Calcula o potencial da distribuição
def pot(charges):
    total = 0
    pot = np.zeros((len(charges)-1,len(charges)-1))
    for i in range(0,len(charges)-1):
        for j in range(i+1,len(charges)-1):
            pot[i][j] = np.log( np.sqrt( (charges[i][0] - charges[j][0])**2 + (charges[i][1] - charges[j][1])**2 ) )
            pot[j][i] = pot[i][j]
            total = total - pot[i][j]
    return total
    
#Move uma carga só
def one_step(charges,a,b,c):
    i = np.random.randint(len(charges)) #sorteia a carga a se mover
    R = np.random.randn() #sorteia uma variavel gaussiana
    step = R*(b)**(1-(charges[i][0]/a)**2-(charges[i][1]/b)**2) #normaliza o tamanho do passo em função da distância à superfície
    n = np.random.randint(4)
    charges_new = charges
    charges_new[i][0] = charges[i][0] + int(step*np.cos(n*(m.pi)/4))
    charges_new[i][1] = charges[i][1] + int(step*np.sin(n*(m.pi)/4))
    
#    pold = pot(charges)
#    pnew = pot(charges_new)
    
    if( ((charges_new[i][0]/a)**2 + (charges_new[i][1]/b)**2 <= 1) & (pot(charges_new) <= pot(charges)) ):
        charges = charges_new
        print(c)
    

#plota distribuição inicial
def plote(charges):
    plt.scatter(charges[:,0],charges[:,1],s=1)
    plt.show()

#main    
charges = initial_condition_np(a,b,n)
plote(charges)
c=1
while c < 5000:
    one_step(charges,a,b,c)
    c = c + 1
plote(charges)
