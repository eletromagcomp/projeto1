import numpy as np
import time
#import scipy as sp
import matplotlib.pyplot as plt
from scipy import stats

def initial_condition_np(a, b, n, n_u, n_v, Inserir_Vies = False): 
       
    charges = np.zeros((n,2), dtype = int)
    charges = np.unique(charges, axis=0) #Exclui cargas sobrepostas
    
    m = len(charges)
    while m!= n:
        
        u = np.random.randint(0, n_u + 1, size = n-m)
        if Inserir_Vies:
            v = custm.rvs(size = n-m) 
        else:
            v = np.random.randint(1, n_v, size = n-m)
    
        elementos_novos = np.zeros((n-m,2))
        elementos_novos[:,0] = u
        elementos_novos[:,1] = v
        charges = np.concatenate((elementos_novos,charges), axis = 0)
        
        charges = np.unique(charges, axis = 0)
        m = len(charges)    
     
    return charges


def Plot(charges, n, c, u_max, n_u, n_v):
    #Esse plot tá meio gambiarra, vou ver como melhorar depois
    
    #Converte para coordenadas cartesianas porque não sei se tem como plotar
    #em coordenadas elípticas com o matplotlib
    coord_cartesianas = np.zeros((n,2), dtype = float)
    coord_cartesianas[:,0] = c * np.cosh(u_max*charges[:,0]/n_u) * np.cos(charges[:,1]*2*np.pi/n_v)
    coord_cartesianas[:,1] = c * np.sinh(u_max*charges[:,0]/n_u) * np.sin(charges[:,1]*2*np.pi/n_v)
    
    #Plota cargas
    plt.scatter(coord_cartesianas[:,0],coord_cartesianas[:,1],s=4)
    
    #Plota grade de coordenadas elípticas
    t = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 100)
    for i in range(n_u):
        plt.plot( c*np.cosh(u_max*i/n_u)*np.cos(t), c*np.sinh(u_max*i/n_u)*np.sin(t), 'k', '--', linewidth = 0.2 )
    for i in range(n_v):
        plt.plot( c*np.cosh(u_max*r)*np.cos(2*np.pi*i/n_v), c*np.sinh(u_max*r)*np.sin(2*np.pi*i/n_v), 'k', '--', linewidth = 0.2 )
    
#    Plota limite da elipse condutora
    plt.plot(c*np.cosh(u_max)*np.cos(t) , c*np.sinh(u_max)*np.sin(t), 'k', linewidth = 0.4 )
#    plt.show()


def distancia(charge1, charge2, i, j, c, n_u, n_v):
    u1, v1 = u_max*charge1[0]/n_u, 2*np.pi*charge1[1]/n_v
    u2, v2 = u_max*charge2[0]/n_u, 2*np.pi*charge2[1]/n_v
    
#    x1, y1 = c * np.cosh(u1) * np.cos(v1), c * np.sinh(u1) * np.sin(v1)
#    x2, y2 = c * np.cosh(u2) * np.cos(v2), c * np.sinh(u2) * np.sin(v2)
    
    dist1 = (np.cosh(2*u1) + np.cos(2*v1))/2
    dist2 = (np.cosh(2*u2) + np.cos(2*v2))/2
    termo_misto = 2*( np.cosh(u1)*np.cos(v1)*np.cosh(u2)*np.cos(v2) + np.sinh(u1)*np.sin(v1)*np.sinh(u2)*np.sin(v2) )
   
    dist = c * np.sqrt( dist1 + dist2 - termo_misto )
    if dist == 0:
        print('################## Dist = 0!! DEU RUIM!!')
        print(dist1, dist2, termo_misto)
        print(dist)
        print(charge1)
        print(charge2)
#    distxy = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )
    
    return dist    


def Potencial_1Carga(charges, i, c, n_u, n_v, Carga_Nova = False, nova_carga = [0,0]):
    carga_central = np.copy(charges[i])
    if Carga_Nova: carga_central = np.copy(nova_carga)
    total = 0
    potencial = np.zeros(len(charges))
    for j in range(len(charges)):
        if j == i:
            potencial[j] = 0
            continue
        potencial[j] = np.log(distancia(carga_central, charges[j], i, j, c, n_u, n_v))
        total = total - potencial[j]
    return total

def Potencial_1Carga_3d(charges, i, c, n_u, n_v, Carga_Nova = False, nova_carga = [0,0]):
    carga_central = np.copy(charges[i])
    if Carga_Nova: carga_central = np.copy(nova_carga)
    total = 0
    potencial = np.zeros(len(charges))
    for j in range(len(charges)):
        if j == i:
            potencial[j] = 0
            continue
        potencial[j] = 1/(distancia(carga_central, charges[j], i, j, c, n_u, n_v))
        total = total + potencial[j]
    return total


def one_step(charges, c, n_u, n_v):
    
    #Flag diz se a carga invadiu a casa de outra
    flag = True
    
    #Sorteio da carga a se mover
    i = np.random.randint(len(charges)) 
    nova_carga = np.copy(charges[i])
    
#    R = np.random.randn() #sorteia uma variavel gaussiana
#    step = R*(b)**(1-(charges[i][0]/a)**2-(charges[i][1]/b)**2) #normaliza o tamanho do passo em função da distância à superfície
    step = 0.1
    
    #Sorteio da direção do passo
    n = np.random.randint(4)
#    print(n)
    
#    print(int(np.cos(n*np.pi/2)), int(np.sin(n*np.pi/2)))
#    print('################################')
#    print(charges[i,0], nova_carga[0])
#    print(charges[i,1], nova_carga[1])
    
    nova_carga[0] = charges[i, 0] + int(step*n_u*np.cos(n*np.pi/2))
    nova_carga[1] = charges[i, 1] + int(step*n_v*np.sin(n*np.pi/2))
    
#    print('################################')
#    print(charges[i,0], nova_carga[0])
#    print(charges[i,1], nova_carga[1])
    
#    if( ( nova_carga[0] <= n_u ) & (nova_carga[0] >= 0 ) & (nova_carga[1] < n_v) & (nova_carga[1] >= 0 ) & (Potencial_1Carga(charges, i, c, Carga_Nova = True, nova_carga = nova_carga) <= Potencial_1Carga(charges, i, c) ) ):
#    if nova_carga in :
#        print('AHA!')
    
    
    for j in range(len(charges)):
        if j == i:
            continue
        if np.array_equal(charges[j], nova_carga):
            flag = False
        if ( (nova_carga[0] == 0) & (charges[j][0] == 0) & (nova_carga[1] + charges[j][1] == 100) ):
            print(nova_carga, charges[j])
            flag = False
            
            
    if flag:
        if ( (0 <= nova_carga[0] <= n_u) and (0 <= nova_carga[1] < n_v) ):
            pot_antigo = Potencial_1Carga(charges, i, c, n_u, n_v)
            pot_novo = Potencial_1Carga(charges, i, c, n_u, n_v, Carga_Nova = True, nova_carga = nova_carga)
            if pot_novo < pot_antigo:
#                print(pot_antigo, pot_novo)
                charges[i] = np.copy(nova_carga)


start = time.time()

a = 100 #Semieixo maior
b = 50 #Semieixo menor
n = 100 #Número de cargas

n_u = 10 #Número de valores possíveis para coordenada u
n_v = 100 #Número de valores possíveis para coordenada v

# Determina foco da elipse
c = np.sqrt( a**2 - b**2 )

#Valor máximo de u - limite da elipse
u_max = np.arccosh(a/c)
#u_max2 = np.arcsinh(b/c)
print('A coordenada u tem limite superior: ' + str(u_max))

#Criar distribuição que tira as cargas das extremidades
xk = np.arange(n_v)
nk = np.zeros(n_v)
nk = 1/(np.abs(np.cos(2*np.pi*xk/n_v))+0.4)
pk = nk / np.sum(nk)
custm = stats.rv_discrete(name='custm', values=(xk, pk))

#Teste da distribuição
#R = custm.rvs(size = 100000)
#plt.hist(R, bins = 100)
#plt.title('Teste do viés de distribuição da coordenada v')
#plt.show()


#Cria condição inicial
charges = initial_condition_np(a, b, n, n_u, n_v, Inserir_Vies = True)

end = time.time()
print('Tempo de simulação (até agora): ' + str(end - start))

Plot(charges, len(charges), c, u_max, n_u, n_v)

passo = 1
while passo < 10000:
    one_step(charges, c, n_u, n_v)
    passo = passo + 1

end = time.time()
print('Tempo de simulação (até agora): ' + str(end - start))

Plot(charges, len(charges), c, u_max, n_u, n_v)

end = time.time()
print('Tempo de simulação (até agora): ' + str(end - start))

#Verifica distribuição das cargas na coordenada v
#plt.hist(charges[:,1], bins = 50)
#plt.show()
#
#plt.hist(u)
#plt.show()
