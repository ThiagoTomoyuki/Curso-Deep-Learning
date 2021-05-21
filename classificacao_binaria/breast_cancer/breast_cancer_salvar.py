# Importando pandas
import pandas as pd
# Importando keras
import keras
# Classe utilizada para criação da rede neural
from keras.models import Sequential
# Importando camadas densas(cada neuronio é ligado a todos os demais na camada oculta) para redes neurais
from keras.layers import Dense, Dropout
# Lendo as variaves de entrada e salvando em previsores
previsores = pd.read_csv('entradas_breast.csv')  
# Lendo as variaves de saida e salvando em classe
classe = pd.read_csv('saidas_breast.csv')  

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
# Salvando os dados em Json
classificador_json = classificador.to_json()
# Salvar no disco
# 'classificador_breast.json' -> nome do arquivo
# 'w' -> comando de escrita
with open('classificador_breast.json','w') as json_file:
    # Escrever na variavel oq é passado, que no caso é um arquivo json chamado classificador_json
    json_file.write(classificador_json)
# Salvando os pesos da rede neural
classificador.save_weights('classificador_breast.h5')