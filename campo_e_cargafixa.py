#%% IMPORT
import time
import numpy as np
import matplotlib.pyplot as plt
#%% VARIÁVEIS
def n(): #Número de cargas
    return 300

def a(): #Semi-eixo maior
    return 100

def b(): #Semi-eixo menor
    return 50


#%% CONDIÇÕES INICIAIS
def initial_condition_np(a,b,n):
    x = np.random.randint(-a,a, size = n)
    lim = np.around(b*np.sqrt(1-(x/a)**2))
    y = np.random.uniform(-1*lim,lim, size = n)#Repara que desse jeito ele gera uma distribuição mais densa nas pontas da elipse
    y = y.astype(int)
    charges = np.zeros((n,2), dtype = int) #Cria array com as posições das cargas
    charges[:,0] = x
    charges[:,1] = y
    charges = np.unique(charges, axis=0) #Exclui cargas sobrepostas    
    return charges


# %% VARIAÇÃO DE POTENCIAL
def Delta_pot(charges, charge_i, charge_i_new, i, condicao):
    #pote: energia de interação da partícula i com a partícula k
    #total: guarda a soma das energias de interações
    #condicao: 1 - carga fixa ; 0 - sem carga fixa
    
    pote_old = np.zeros(1) #Tamanho 1 porquê entra só a coordenada inicial da partícula i
    pote_new = np.zeros(4) #Tamanho 4 porquê entra as coordenadas dos 4 movimentos possíveis da partícula i (norte, sul, leste, oeste)
    total_old = np.zeros(1)
    total_new = np.zeros(4)
    
    if condicao == 1:
        carga_fixa = [a+20,b+20]
        total_old = -(len(charges)/2)*np.log(np.sqrt(np.sum((charge_i - carga_fixa)**2)))
        total_new = -(len(charges)/2)*np.log(np.sqrt(np.sum((charge_i_new - carga_fixa)**2, axis = 1)))
        
    for k in range(len(charges)):
        if i!=k: #Não calculamos a energia de interação de uma partícula com ela mesma
            charge_k = charges[k, :]
            pote_old = np.log(np.sqrt(np.sum((charge_i - charge_k)**2)))
            pote_new = np.log(np.sqrt(np.sum((charge_i_new - charge_k)**2, axis = 1)))
            total_old = total_old- pote_old
            total_new = total_new- pote_new
    Delta_U = total_new - total_old
    min_index = np.argmin(Delta_U) #Pega o primeiro índice do menor Delta_U
    Delta_U_min = Delta_U[min_index] #Pega o valor do menor Delta_U
    return Delta_U_min, min_index



#%% SIMULAÇÃO
def simulate(a, b, n, charges, condicao):
    variacao = np.array([[1,0], [-1,0], [0,1], [0,-1]]) #Matriz de variação, contém os movimentos possíveis (norte, sul, leste, oeste)
    index_charges = np.arange(len(charges))
    flag = 0
    while flag!=int(len(charges)): #Cada iteração é um ciclo que roda por todas as partículas
        flag = 0
        np.random.shuffle(index_charges) #Fahttps://www.facebook.com/z um shuffle nos index_charges, para que simulemos a escolha aleatória das partículas
        for i in index_charges:
            charges_new = np.copy(charges) 
            charge_i = np.copy(charges[i, :]) #Pego a partícula com indice i
            
            R = abs(np.random.randn()) + 1 #Gera uma variável gausiana aleatória
            elipse = (charges[i][0]/a)**2+(charges[i][1]/b)**2 #x²/a² + y²/b²
            step = int(R*(b)**(1- elipse)) #Tamanho do passo, no que quanto mais proximo da elipse, o expoente se aproxima de 0
            
            charge_i_new = charge_i + step*variacao #Aplico a variação para todas as direções possíveis para a partícula i
            for I in range(4): #Aqui quero ver se o movimento que eu to realizando é permitido, ou seja, se a partícula cai pra fora da elipse ou ocupa o lugar de outra carga
                charges_new[i, :] = charge_i_new[I, :] 
                elipse = (charge_i_new[I, 0]/a)**2 + (charge_i_new[I, 1]/b)**2 #x²/a² + y²/b²
                if(elipse <= 1) and (len(np.unique(charges_new, axis=0)) == len(charges)):
                    continue
                else:
                    charge_i_new[ I, :] = charge_i_new[I, :] - step*variacao[I, :] #Caso a partícula não satisfaça as duas condições, volto ela pra sua posição inicial
                    #Note que voltar pra posição inicial implica que o Delta_U para esse movimento vai ser 0 
            Delta_U_min, min_index = Delta_pot(charges, charge_i, charge_i_new, i, condicao) #Vou pegar a menor variação energia dentre os movimentos e o índice que causa essa variação
            if Delta_U_min<0:
                charges[i, :] = charge_i + step*variacao[min_index, :] #Movo a partícula pra a direção que causa maior diminuição da energia
            elif Delta_U_min==0:
                flag = flag + 1 #Se a menor variação de energia da partícula é nula, contamos 1 a mais no flag
                #Note que o flag reinicia a cada ciclo
                #Se o número de partículas cuja menor variação de energia é 0 for igual ao número de partículas, alcançamos um mínimo local.
    return charges


#%% CAMPO ELETRICO
def campo_eletrico(a,b,n,charges):
    ponto = np.array([np.random.randint(-a/2,a/2),np.random.randint(-b/2,b/2)])
    campo = 0
    for k in range(len(charges)):
        charge_k = charges[k, :]
        campo = campo + (np.sum((ponto - charge_k)))
        
    campo = (n*1.44*10**(9))*campo #Correção com constante e cargas
    
    print(1/campo,ponto)
    return campo
#%% PLOT
def plot(a, b, charges_0, charges):
    #Copia as posições iniciais em x e y
    x_0 = charges_0[:, 0]
    y_0 = charges_0[:, 1]
        
    #Copia as posições finais em x e y
    x = charges[:, 0]
    y = charges[:, 1]
    
    #Define a superfície da elipse
    x_superficie = np.linspace(-a, a, 1000) #Para gerar a superfície
    #y_superficie = b*np.sqrt(1-(x_superficie/a)**2)
    
    plt.axes().set_aspect('equal') #Mesmo aspect ratio para que vejamos a deformação da elipse
    plt.grid()
    plt.title("Condutor elíptico bidimensional")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x_superficie, y_superficie, color='yellow') #Superfície
    plt.plot(x_superficie, -y_superficie, color='yellow') #Superfície
    plt.plot(a+20,b+20,marker='o',color='pink')
    plt.scatter(x_0, y_0, s=3, color='red', label = 'Inicial') #Inicial
    plt.scatter(x, y, s=3, color='darkblue', label ='Final') #Final
    plt.legend(loc=4)
    plt.savefig('plot.pdf')
    plt.show()


#%% CÁLCULOS (LARGE STEP)

a = a()
b = b()
n = n()

condicao = 1

start = time.time()

charges = initial_condition_np(a,b,n)
charges_0 = np.copy(charges)
charges = simulate(a,b,n, charges,condicao)
campo = campo_eletrico(a,b,n,charges)

end = time.time()
print('Tempo de simulação: ' + str(end - start))


#%% PLOT (LARGE STEP)
plot(a, b, charges_0, charges)
