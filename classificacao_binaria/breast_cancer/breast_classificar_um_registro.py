# Importando pandas
import pandas as pd
# Importando keras
import keras
# Importando o numpy
import numpy as np
# Importando camadas densas(cada neuronio é ligado a todos os demais na camada oculta) para redes neurais
from keras.layers import Dense, Dropout
# Importando Sequential
from keras.models import Sequential
# Lendo as variaves de entrada e salvando em previsores
previsores = pd.read_csv('entradas_breast.csv')  
# Lendo as variaves de saida e salvando em classe
classe = pd.read_csv('saidas_breast.csv')  

# Colocando os melhores valores encostrados no tuning 

# Criando rede neural
classificador = Sequential()
# Adicionando a primeira camada oculta
classificador.add(Dense(units = 8, activation='relu',kernel_initializer='normal', input_dim = 30))
# Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
classificador.add(Dropout(0.2))
# Adicionando mais uma camada oculta
classificador.add(Dense(units = 8, activation='relu' ,kernel_initializer='normal'))
# Zerando mais 20% dos valores da camada oculta
classificador.add(Dropout(0.2))
# Fazendo a camada de saida
classificador.add(Dense(units=1,activation='sigmoid'))

# Compilando a rede neural
classificador.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['binary_accuracy',])
classificador.fit(previsores,classe,batch_size=10,epochs=100)
# Novo registro
novo = np.array([[15.80, 8.34, 118, 900, 0.1, 0.26, 0.08, 0.134, 0.178,
                  0.2, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.3, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
# Devolve se tem(1) ou n(0) cancer
previsao = classificador.predict(novo)
previsao = (previsao>0.5)