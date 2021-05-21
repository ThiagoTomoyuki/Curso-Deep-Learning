# Importando pandas
import pandas as pd
# Importando keras
import keras
# Classe utilizada para criação da rede neural
from keras.models import Sequential
# Importando camadas densas(cada neuronio é ligado a todos os demais na camada oculta) para redes neurais
from keras.layers import Dense, Dropout
# Importando wrappers do scikit_learn
from keras.wrappers.scikit_learn import KerasClassifier
# Importando GridSearchCV -> para fazer pesquisa em grades
from sklearn.model_selection import GridSearchCV
# Lendo as variaves de entrada e salvando em previsores
previsores = pd.read_csv('entradas_breast.csv')  
# Lendo as variaves de saida e salvando em classe
classe = pd.read_csv('saidas_breast.csv')  
# funcao q cria a rede neural
def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    # Criando rede neural
    classificador = Sequential()
    # Adicionando a primeira camada oculta
    # Units - quantos neuronios faram parte da camada oculta
    # Calculo do Units ((30+1)/2) -> ((numero de entradas+n de neuronios na camada de saida)/2) -> arrredondando para mais
    # Activation -> funcao de ativação 
    # kernel_initializer -> Inicializacao dos pesos 
    # input_dim ->  quantidade de elementos q tem na camada de entrada
    classificador.add(Dense(units = neurons, activation=activation,kernel_initializer=kernel_initializer, input_dim = 30))
    # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    classificador.add(Dropout(0.2))
    # Adicionando mais uma camada oculta
    classificador.add(Dense(units = neurons, activation=activation ,kernel_initializer=kernel_initializer))
    # Zerando mais 20% dos valores da camada oculta
    classificador.add(Dropout(0.2))
    # Fazendo a camada de saida
    classificador.add(Dense(units=1,activation='sigmoid'))
    
    # Compilando a rede neural
    # Optimizer -> funcao de ajuste dos pesos
    # loss -> funcao de perda(tratar erro)
    # metrics -> Metrica utilizada para avaliação
    classificador.compile(optimizer= optimizer,loss=loos, metrics=['binary_accuracy',])
    # Retorna o cassificador
    return classificador

# Criando rede neural
classificador = KerasClassifier(build_fn=criarRede)
# Recebe um dicionario que tem as varias formas que vc escolhe para ver o melhor resultado
parametros = {'batch_size':[10,30],
              'epochs':[50, 100],
              'optimizer': ['adam','sgd'],
              'loos':['binary_crossentropy', 'hinge'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu','tanh'],
              'neurons':[16,8]}
# Fazer a busca
grid_search = GridSearchCV(estimator=classificador,
                           param_grid=parametros,
                           scoring='accuracy',
                           cv=5)
# Treinamento da rede
grid_search = grid_search.fit(previsores,classe)
# Retorna os melhores valores da lista de parametros
melhores_parametros = grid_search.best_params_
# Pega os melhores parametros passados a cima e cria efetivamente a rede neural
melhor_precisao = grid_search.best_score_


