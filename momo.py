import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats


a = 100
b = 50
n = 100 #Número de cargas
n_u = 10 #
n_v = 100

c = np.sqrt( a**2 - b**2 )

#Valor máximo de u - limite da nossa elipse
u_max = np.arccosh(a/c)
#u_max2 = np.arcsinh(b/c)
print(u_max)

#Criar distribuição que tira as cargas das extremidades
xk = np.arange(n_v)+1
nk = np.zeros(n_v)
nk = 1/(np.abs(np.cos(2*np.pi*xk/n_v))+0.1)
pk = nk / np.sum(nk)
custm = stats.rv_discrete(name='custm', values=(xk, pk))
R = custm.rvs(size = 100000)
plt.hist(R, bins = 100)
plt.show()

def initial_condition_np(a, b, n, n_u, n_v):

    u = np.random.randint(0, n_u + 1, size = n)
    #v = np.random.randint(1, n_v + 1, size = n)
    v = custm.rvs(size = n)
    charges = np.zeros((n,2), dtype = int)
    
    charges[:,0] = u
    charges[:,1] = v
    
    charges = np.unique(charges, axis=0) #Exclui cargas sobrepostas
    
    m = len(charges)
    while m!= n:
        u = np.random.randint(1, n_u + 1, size = n - m)
        v = custm.rvs(size = n - m)
    
        elementos_novos = np.zeros((n-m,2))
        elementos_novos[:,0] = u
        elementos_novos[:,1] = v
        charges = np.concatenate((elementos_novos,charges), axis = 0)
        
        charges = np.unique(charges, axis = 0)
        m = len(charges)    
     
    return charges

def distancia(charge1, charge2, c):
    u1, v1 = u_max*charge1[0]/n_u, 2*np.pi*charge1[1]/n_v
    u2, v2 = u_max*charge2[0]/n_u, 2*np.pi*charge2[1]/n_v
    
    dist1 = (np.cosh(2*u1) + np.cos(2*v1))/2
    dist2 = (np.cosh(2*u2) + np.cos(2*v2))/2
    termo_misto = -2*( np.cosh(u1)*np.cos(v1)*np.cosh(u2)*np.cos(v2) + np.sinh(u1)*np.sin(v1)*np.sinh(u2)*np.sin(v2) )
    
    dist = c * np.sqrt( dist1 + dist2 + termo_misto )
    
    x1 = c * np.cosh(u1) * np.cos(v1)
    y1 = c * np.sinh(u1) * np.sin(v1)
    
    x2 = c * np.cosh(u2) * np.cos(v2)
    y2 = c * np.sinh(u2) * np.sin(v2)
    
    print(np.sqrt((x1-x2)**2+(y1-y2)**2))
    
    return dist

def plote(charges, n, c, u_max, n_u, n_v):
    #Esse plot tá meio gambiarra, vou ver como melhorar depois
    
    #Converte para coordenadas cartesianas porque não sei se tem como plotar
    #em coordenadas elípticas com o matplotlib
    coord_cartesianas = np.zeros((n,2), dtype = float)
    coord_cartesianas[:,0] = c * np.cosh(u_max*charges[:,0]/n_u) * np.cos(charges[:,1]*2*np.pi/n_v)
    coord_cartesianas[:,1] = c * np.sinh(u_max*charges[:,0]/n_u) * np.sin(charges[:,1]*2*np.pi/n_v)
    
    #Plota cargas
    plt.scatter(coord_cartesianas[:,0],coord_cartesianas[:,1],s=1)
    
    #Plota grade de coordenadas elípticas
    t = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 100)
    for i in range(n_u):
        plt.plot( c*np.cosh(u_max*i/n_u)*np.cos(t), c*np.sinh(u_max*i/n_u)*np.sin(t), 'k', '--', linewidth = 0.2 )
    for i in range(n_v):
        plt.plot( c*np.cosh(u_max*r)*np.cos(2*np.pi*i/n_v), c*np.sinh(u_max*r)*np.sin(2*np.pi*i/n_v), 'k', '--', linewidth = 0.2 )
    
    #Plota limite da elipse condutora
    plt.plot(c*np.cosh(u_max)*np.cos(t) , c*np.sinh(u_max)*np.sin(t), 'k' )
    plt.show()
    plt



charges = initial_condition_np(a, b, n, n_u, n_v)
    
plote(charges, len(charges), c, u_max, n_u, n_v)

#Verifica distribuição das cargas na coordenada v
plt.hist(charges[:,1], bins = 50)
plt.show()
#
#plt.hist(u)
#plt.show()
print(charges[0,:])
print(charges[1,:])
print(distancia(charges[0,:], charges[1,:], c))