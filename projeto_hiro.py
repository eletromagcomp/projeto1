#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:35:30 2019

@author: hiro
"""


import time
import numpy as np
import matplotlib.pyplot as plt
############################################################################################
def n(): #Número de cargas
    return 500

def a(): #Semi-eixo maior
    return 100

def b(): #Semi-eixo menor
    return 50
#########################################################################################
def initial_condition_np(a,b,n):
    x = np.random.randint(-a,a, size = n)
    lim = np.around(b*np.sqrt(1-(x/a)**2))
    y = np.random.uniform(-1*lim,lim, size = n)#|^|repara que desse jeito ele gera uma distribuição mais densa nas pontas da elipse
    y = y.astype(int)
    #cria array com as posições das cargas
    charges = np.zeros((n,2), dtype = int)
    charges[:,0] = x
    charges[:,1] = y
    charges = np.unique(charges, axis=0) #Exclui cargas sobrepostas    
    return charges
##############################################################
def Delta_pot(charges, charge_i, charge_i_new, i):
    total_old = np.zeros(1)
    total_new = np.zeros(4)
    pote_old = np.zeros(1)
    pote_new = np.zeros(4)
    for k in range(len(charges)):
        if i!=k:
            charge_k = charges[k, :]
            pote_old = np.log(np.sqrt(np.sum((charge_i - charge_k)**2)))
            pote_new = np.log(np.sqrt(np.sum((charge_i_new - charge_k)**2, axis = 1)))
            total_old = total_old- pote_old
            total_new = total_new- pote_new
    Delta_U = total_new - total_old
    min_index = np.argmin(Delta_U)
    Delta_U_min = Delta_U[min_index]
    return Delta_U_min, min_index
#########################SIMULAÇÃO###########################
a = a()
b = b()
n = n()
start = time.time()

charges = initial_condition_np(a,b,n)

x_0 = np.copy(charges[:, 0])
y_0 = np.copy(charges[:, 1])
##################################################################################################
variacao = np.array([[1,0], [-1,0], [0,1], [0,-1]])
index_charges = np.arange(len(charges))
for iteracao in range(100):
    np.random.shuffle(index_charges)
    for i in index_charges:
        charges_new = np.copy(charges)
        charge_i = np.copy(charges[i, :])
        charge_i_new = charge_i + variacao
        for I in range(4):
            charges_new[i, :] = charge_i_new[I, :]
            elipse = (charge_i_new[I, 0]/a)**2 + (charge_i_new[I, 1]/b)**2
            if(elipse <= 1) and (len(np.unique(charges_new, axis=0)) == len(charges)):
                continue
            else:
                charge_i_new[ I, :] = charge_i_new[I, :] - variacao[I, :]
                
        Delta_U_min, min_index = Delta_pot(charges, charge_i, charge_i_new, i)
        if Delta_U_min<0:
            charges[i, :] = charge_i + variacao[min_index, :]
    print(iteracao)
x = charges[:, 0]
y = charges[:, 1]
########################PLOT################################
x_superficie = np.linspace(-a, a, 1000) #Para gerar a superfície
y_superficie = b*np.sqrt(1-(x_superficie/a)**2)

plt.axes().set_aspect('equal')
plt.title("Distribuição inicial")
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_superficie, y_superficie, color='black')
plt.plot(x_superficie, -y_superficie, color='black')
plt.scatter(x, y, s=5, color='darkblue')
plt.scatter(x_0, y_0, s=5, color='red')
plt.show()



end = time.time()

print('Tempo de simulação (até agora): ' + str(end - start))


