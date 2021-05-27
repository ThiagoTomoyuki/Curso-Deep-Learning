# AUMENTA O NUMERO DE IMAGENS

# Mnist onde contem a base de dados
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
# Importando as 4 etapas do processo convolucional
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten

# Carregando base de dados
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Transformando os dados 
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

# 'Normalizando' os dados para que eles fiquem nos valores entre 0 - 1
previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# Criando o classificador
classificador = Sequential()
# Primeira etapa - Operador de convolução
# Conv2D(quantidade de filtros, tamanho do dector de caracteristicas, tamanho da imagem, funcao de ativacao)
classificador.add(Conv2D(32, (3,3),input_shape=(28, 28, 1),activation = 'relu'))
# Segunda etapa - Pooling
classificador.add(MaxPool2D(pool_size=(2,2)))
# Terceira etapa - Flatteninig
classificador.add(Flatten())
# Gerando a rede neural
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])

# Aumentando a quantidade de imagens

# ImageDataGenerator(rotation_range=rotação da imagem,horizontal_flip=giros horizontais na imagem,shear_range=altera valores dos pixeis,height_shift_range=faz a modificacao na altura da image,zoom_range=modifica o zoom da imagem)
gerador_treinamento = ImageDataGenerator(rotation_range=7,horizontal_flip=True,shear_range=0.2,height_shift_range=0.07,zoom_range=0.2)
gerador_teste = ImageDataGenerator()
# Gera base de dados de treinamento
base_treinamento = gerador_treinamento.flow(previsores_treinamento,classe_treinamento,batch_size=128)
# Gera base de dados de teste
base_teste = gerador_teste.flow(previsores_teste,classe_teste,batch_size=128)

classificador.fit_generator(base_treinamento,steps_per_epoch=60000/128,epochs=5,validation_data=base_teste,validation_steps=10000/128)