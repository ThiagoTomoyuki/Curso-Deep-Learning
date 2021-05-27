# Biblioteca especifica para vizualizacao de dados
import matplotlib.pyplot as plt
# Mnist onde contem a base de dados
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
# Importando as 4 etapas do processo convolucional
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten

# Carregando base de dados
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()
# Vizualizando as imagens que foram carregadas(um exemplos)
plt.imshow(X_treinamento[0],cmap = 'gray')

# Converter dados

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
# Normalizando as camadas de convolucao
classificador.add(BatchNormalization())
# Segunda etapa - Pooling
classificador.add(MaxPool2D(pool_size=(2,2)))

## Terceira etapa - Flatteninig
# Nao pode adicionar Flatten quando se tem mais de uma camada. So se pode adicionar o Flatten no final 
#classificador.add(Flatten())

# Adicionando mais uma camada de convolução
classificador.add(Conv2D(32, (3,3),activation = 'relu'))
# Normalizando as camadas de convolucao
classificador.add(BatchNormalization())
# Segunda etapa - Pooling
classificador.add(MaxPool2D(pool_size=(2,2)))
# Terceira etapa - Flatteninig
classificador.add(Flatten())
# Gerando a rede neural
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=128, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units=10, activation='softmax'))
classificador.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,batch_size = 128, epochs = 5,validation_data = (previsores_teste, classe_teste))

resultado = classificador.evaluate(previsores_teste, classe_teste)

