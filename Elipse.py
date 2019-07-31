#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:43:49 2019

@author: lordemomo
"""

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import random


def Criar_Condicao_Inicial(a, b, n):
#Essa função cria uma malha retangular a por b e distribui n cargas em uma 
#elipse inserida na malha (com semieixos a e b). Os valores de retorno são as variáveis Charge
#(coordenadas de cada carga) e r (distâncias entre cada carga).
    
    Charges = [] #Lista que gardará as coordenadas de cada carga
    rj = [] #Variável intermediária para guardar distâncias
    r = [] #Lista com distâncias finais entre cada carga
    
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
#        print(Charges[i]) #Mostrar a posição de todas as cargas
        
    print('############### Cargas distribuídas!')
    
    for i in range(0,n):
        for j in range(0,n):
            r_ij = np.sqrt((Charges[i][0]-Charges[j][0])**2+(Charges[i][1]-Charges[j][1])**2)
            rj.append(r_ij)
        r.append(rj)
    
    print('############### Distâncias calculadas!')
#    print(Charges[0], Charges[1]) #Verificar cálculo da distância
#    print(r[0][1])
    
    
    #Plot simples da condição inicial
    plt.scatter(x_vect, y_vect, s=10, alpha=0.5)
    plt.show()
    
    return Charges, r
