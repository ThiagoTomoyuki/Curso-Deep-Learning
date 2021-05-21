import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
# Lendo o arquivo de iris.csv e colocando na variavel base
base = pd.read_csv('iris.csv')
# separando os previsores e a clase do arquivo todo(ires.csv)
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:,4].values
# Importando LabelEncoder -> Transformar de atributo categorico para atributo numerico
from sklearn.preprocessing import LabelEncoder
# Criando e transformando os valores que eram String para valores numericos(categoricos)
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
# Iris setosa       1 0 0
# Iris virginica    0 1 0
# Iris versicolor   0 0 1
# Mudando a variavel simples para um vetor de 3 colunas -> Para identificar o tipo de Iris
classe_dummy = np_utils.to_categorical(classe)
# Importando train_test_split para fazer teste e treinamento da rede 
from sklearn.model_selection import train_test_split
# Criando base de dados de treinamento e de teste
# test_size -> valor passado é igual a porcentagem de dados usados para test
previsores_treinamento,previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)
# Construcao da estrutura da rede neural
classificador = Sequential()
# Adicionando a primeira camada
# Units - quantos neuronios faram parte da camada oculta
# Calculo do Units ((4+3)/2) -> ((numero de entradas+n de neuronios na camada de saida)/2) -> arrredondando para mais
classificador.add(Dense(units=4,activation='relu',input_dim=4))
# Adicionando mais uma camada
classificador.add(Dense(units=4,activation='relu'))
# Adicionando a camada de saida
# Quando o problema exige mais de duas classificacoes de classes -> utilizar a funcao de ativacao 'softmax'
classificador.add(Dense(units=3,activation='softmax'))
# Compilando a rede neural
classificador.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
# Fazendo o treinamento
classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10,epochs=1000)
# Metodo especifico do keras q faz a avaliacao automatica
resultado = classificador.evaluate(previsores_teste,classe_teste)
# Vizualizacao da matriz de confusao
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5)
import numpy as np
# Passando os valores de vetores de arrays para numeros
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2=[np.argmax(t) for t in previsoes]
# Importar confusion_matrix -> Para ter acecsso a matriz de confusao
from sklearn.metrics import confusion_matrix
# Fazer matriz de confusao -> Para analisar os resultados
matriz = confusion_matrix(previsoes2,classe_teste2)
