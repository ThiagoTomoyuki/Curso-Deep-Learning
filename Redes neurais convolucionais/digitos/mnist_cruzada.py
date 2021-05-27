
# Mnist onde contem a base de dados
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold
# Importando as 4 etapas do processo convolucional
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten

seed = 5
# Muda a semente geradora dos numeros aleatorios
np.random.seed(seed)
# Carregando base de dados
(X, y), (X_teste, y_teste) = mnist.load_data()
# Transformando os dados 
previsores = X.reshape(X.shape[0],28, 28, 1)
previsores = previsores.astype('float32')

# 'Normalizando' os dados para que eles fiquem nos valores entre 0 - 1
previsores /= 255

classe = np_utils.to_categorical(y, 10)
# kfold controla a validação 
# StratifiedKFold(n_splits=numero de fold, shuffle=True(pegar dados aleatoriamente), random_state=seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
# resultados - recebe os resultados de cada execucao
resultados = []

a = np.zeros(5)
b = np.zeros(shape=(classe.shape[0],1))

# Percorre todos os registros
for indice_treinamento , indice_teste in kfold.split(previsores,np.zeros(shape=(classe.shape[0],1))):
    #print('Indices treinamento',indice_treinamento,'Indice teste', indice_teste)
    # Criando o classificador
    classificador = Sequential()
    # Primeira etapa - Operador de convolução
    # Conv2D(quantidade de filtros, tamanho do dector de caracteristicas, tamanho da imagem, funcao de ativacao)
    classificador.add(Conv2D(32, (3,3),input_shape=(28, 28, 1),activation = 'relu'))
    # Segunda etapa - Pooling
    classificador.add(MaxPool2D(pool_size=(2,2)))
    # Terceira etapa - Flatteninig
    classificador.add(Flatten())
    classificador.add(Dense(units=128, activation='relu'))
    classificador.add(Dense(units=10, activation='softmax'))
    classificador.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento], batch_size=128, epochs=5)
    precisao = classificador.evaluate(previsores[indice_teste],classe[indice_teste])
    resultados.append(precisao[1])
