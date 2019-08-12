#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:35:30 2019

@author: hiro
"""


import time
import numpy as np
import matplotlib.pyplot as plt
import random
   
def inital_condition_np(a, b, n):
#Gerando uma malha quadrada com uma elipse dentro fazendo a distribuição de cargas  
    x = np.random.randint(-a,a, size = n) #Gerar coordenadas de cada carga no eixo x
    
    lim_inf = np.around(-b*np.sqrt(1-(x/a)**2)) #Restrição numa elipse
    lim_sup = np.around(b*np.sqrt(1-(x/a)**2))
    y = np.random.uniform(lim_inf, lim_sup, size = n)
    y = y.astype(int)
    
    #Garante que duas cargas não caiam na mesma posição
    xy = np.zeros((n, 2)) #Quero criar um array onde cada linha uma carga e as colunas são o eixo x e y
    xy[:, 0] = x
    xy[:, 1] = y
    elementos_unicos = np.unique(xy, axis = 0) #Gera apenas com as coordenadas (x,y) que aparecem uma unica vez
    m = len(elementos_unicos[:]) 
    while m!= n:
        #Geramos novas coordenadas para substituir as repetidas e 
        #repetimos isso até não houver mais repetidos
        x = np.random.randint(-a,a, size = n - m) 
    
        lim_inf = np.around(-b*np.sqrt(1-(x/a)**2))
        lim_sup = np.around(b*np.sqrt(1-(x/a)**2))
    
        y = np.random.uniform(lim_inf, lim_sup, size= n - m)
        y = y.astype(int)
        
        elementos_novos = np.zeros((n-m,2))
        elementos_novos[:,0] = x
        elementos_novos[:,1] = y
        xy = np.concatenate((elementos_novos,elementos_unicos), axis = 0)
        
        elementos_unicos = np.unique(xy, axis = 0)
        m = len(elementos_unicos[:])
    
    #Extraindo um array para cada um dos eixos
    x = elementos_unicos[:,0]
    y = elementos_unicos[:,1]
    return x, y



def initial_condition_py(a, b, n): #Código do Momo
#Essa função cria uma malha retangular a por b e distribui n cargas em uma 
#elipse inserida na malha (com semieixos a e b). Os valores de retorno são as variáveis Charge
#(coordenadas de cada carga) e r (distâncias entre cada carga).
    Charges = [] #Lista que gardará as coordenadas de cada carga

    
    x_vect = []
    y_vect = []
    
    
    for i in range(0,n):
        x = random.randint(-a,a)
        y = random.randint(int(-b*np.sqrt(1-(x/a)**2)),int(b*np.sqrt(1-(x/a)**2))) #Restrição da elipse
        
        x_vect.append(x)
        y_vect.append(y)
    
    #Garante que duas cargas não caiam na mesma posição
        for interacao in range(0,100):
            todas_diferentes = 1
            for j in range(0,i):
                if x_vect[j] == x_vect[i] & y_vect[j] == y_vect[i]:
                    x = random.randint(-a,a)
                    y = random.randint(int(-b*np.sqrt(1-(x/a)**2)),int(b*np.sqrt(1-(x/a)**2)))
                    x_vect[i] = x
                    y_vect[i] = y
                    todas_diferentes = 0
            if todas_diferentes:
                break
                    
        Charges.append([x,y])
    return x_vect, y_vect
##############################################################
def distance_charges_np(xy):
    #Calcular a distância entre as cargas.
    
    #A ideia é gerar uma matriz n x n em que cada entrada é 
    #a distância de interação da carga i com a carga j,
    #com i, j de 1 a n
    
    #Da forma que faço aqui, o elemento (i, j) da matriz não
    #corresponde com a distancia entre i e j, mas isso não importa.
    r = np.zeros((n, n-1))
    xy_aux = np.copy(xy) 
    
    #Faremos permutações cíclicas em xy_aux mantendo xy imutável
    #Se tivermos xy_aux como:
    # 1 10
    # 2 20
    # 3 30
    # 4 40
    #O que vamos fazer é criar
    # 4 40
    # 1 10
    # 2 20
    # 3 30
    #Assim, podemos calcular as distancias de todas as combinações (i,j)
    for iteracao in range(n-1):
        xy_ultimo = np.copy(xy_aux[-1,:]) #Coordenada da última carga
        xy_ultimo = np.array([xy_ultimo]) #Fazendo isso para usar o concatenate
        xy_del = np.delete(xy_aux, -1, axis= 0) #Excluindo a ultima linha em xy_aux
        xy_aux = np.concatenate((xy_ultimo, xy_del), axis = 0) #Inserindo as coordenadas da última carga no início de xy_aux
        r_ij = (xy - xy_aux)**2 #Calculando (x - x_aux)^2 e (y - y_aux)^2
        r_ij = np.sqrt(np.sum(r_ij, axis= 1)) #sqrt((x - x_aux)^2 + (y - y_aux)^2)
        r[:, iteracao] = r_ij #Inserindo na matriz r
    
    return r
#########################SIMULAÇÃO###########################
a = 100 #Semi eixo maior
b = 50 #Semi eixo menor
n = 100 #Número de cargas
start = time.time()

x, y = inital_condition_np(a,b,n)


xy = np.zeros((n, 2)) 
xy[:, 0] = x
xy[:, 1] = y
##################################################################################################
xy_novo = np.copy(xy)
variacao = np.zeros(2)
for i in range(10000):
    direcao = np.random.choice(np.array((False,True)), 2, replace=False) 
    sentido  = np.random.choice(np.array((-1,1))) 
    variacao[0] = np.where(direcao[0]==True, sentido, 0)
    variacao[1] = np.where(direcao[1]==True, sentido, 0)
    index_mover = np.random.randint(n)
    xy_novo[index_mover, :] = xy[index_mover, :] + variacao
    unicos = np.unique(xy_novo, axis = 0)
    m = len(unicos[:])
    #JUNTAR TODOS OS IFS
    if m!=n:
        xy_novo[index_mover, :] = xy_novo[index_mover, :] - variacao
    else:
        elipse = (xy_novo[index_mover, 0]/a)**2 + (xy_novo[index_mover, 1]/b)**2
        if elipse>1:
            xy_novo[index_mover, :] = xy_novo[index_mover, :] - variacao
        else:
            r_velho = distance_charges_np(xy)
            r_novo = distance_charges_np(xy_novo)
            Delta_U = - (np.log(r_novo) - np.log(r_velho))
            Delta_U = np.sum(Delta_U)
            if Delta_U>0:
                 xy_novo[index_mover, :] = xy_novo[index_mover, :] - variacao
    xy = np.copy(xy_novo)


x_novo = xy[:, 0]
y_novo = xy[:, 1]
########################PLOT################################
x_superficie = np.linspace(-a, a, 1000) #Para gerar a superfície
y_superficie = b*np.sqrt(1-(x_superficie/a)**2)

plt.title("Distribuição inicial")
plt.xlabel('x')
plt.ylabel('y')
plt.axes().set_aspect('equal')
plt.plot(x_superficie, y_superficie, color='black')
plt.plot(x_superficie, -y_superficie, color='black')
plt.scatter(x, y, s=5, color='darkblue')
plt.scatter(x_novo, y_novo, s=5, color='red')
plt.savefig('distribuicao_inicial.pdf')
plt.show()



end = time.time()

print('Tempo de simulação (até agora): ' + str(end - start))


