import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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
# Funcao q cria rede
def criar_Rede(optimizer, loos, kernel_initializer, activation, neurons):
    # Construcao da estrutura da rede neural
    classificador = Sequential()
    # Adicionando a primeira camada
    # Units - quantos neuronios faram parte da camada oculta
    # Calculo do Units ((4+3)/2) -> ((numero de entradas+n de neuronios na camada de saida)/2) -> arrredondando para mais
    classificador.add(Dense(units=neurons,activation=activation,input_dim=4))
    classificador.add(Dropout(0.2))
    # Adicionando a camada de saida
    # Quando o problema exige mais de duas classificacoes de classes -> utilizar a funcao de ativacao 'softmax'
    classificador.add(Dense(units=3,activation='softmax'))
    # Compilando a rede neural
    classificador.compile(optimizer=optimizer,loss=loos,metrics=['categorical_accuracy'])
    return classificador
# Criar a rede
classificador = KerasClassifier(build_fn=criar_Rede)
# Recebe um dicionario que tem as varias formas que vc escolhe para ver o melhor resultado
parametros = {'batch_size':[10,30],
              'epochs':[1000, 1500],
              'optimizer': ['adam','sgd'],
              'loos':['categorical_crossentropy', 'sparse_categorical_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation':['relu','softmax','tanh'],
              'neurons':[4,8,3]}
import numpy as np
# Passando os valores de vetores de arrays para numeros
classe2 = [np.argmax(t) for t in classe_dummy]
previsoes2=[np.argmax(t) for t in previsores]
# Fazer a busca
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           cv=5)
# Treinamento da rede
grid_search = grid_search.fit(previsoes2,classe2)
# Retorna os melhores valores da lista de parametros
melhores_parametros = grid_search.best_params_
# Pega os melhores parametros passados a cima e cria efetivamente a rede neural
melhor_precisao = grid_search.best_score_

####################################################################################################################################

# Sem mexer em nada a media = 0.95
# Tirando uma camada oculta media = 0.96