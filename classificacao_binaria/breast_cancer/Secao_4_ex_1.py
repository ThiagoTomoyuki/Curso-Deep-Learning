import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')  
classe = pd.read_csv('saidas_breast.csv')  

def criarRede():
    # # Criando rede neural
    # classificador = Sequential()
    # # Adicionando a primeira camada oculta
    # # Units - quantos neuronios faram parte da camada oculta
    # # Calculo do Units ((30+1)/2) -> ((numero de entradas+n de neuronios na camada de saida)/2) -> arrredondando para mais
    # # Activation -> funcao de ativação 
    # # kernel_initializer -> Inicializacao dos pesos 
    # # input_dim ->  quantidade de elementos q tem na camada de entrada
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    
    #######################################################################################################################
    
    # # PARTE 1:
    # #Teste de adicionar mais camadas para ver se melhora a precisao
    # # Adicionando a primeira camada oculta
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # # Adicionando a primeira camada oculta
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    
    #######################################################################################################################
    
    # #PARTE 2:
    ##Mudar o numero de neuronios para 32
    # classificador.add(Dense(units = 32, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 32, activation='relu',kernel_initializer='random_uniform'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    # #Mudar a quantidade de neuronios
    # classificador.add(Dense(units = 16, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    #######################################################################################################################
    
    # # PARTE 3:
    # # Mudando a quantidade de neuronios zerados
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.3 -> zera 30% dos valores
    # classificador.add(Dropout(0.3))
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform'))
    # # Zerando mais 30% dos valores da camada oculta
    # classificador.add(Dropout(0.3))
    
    #######################################################################################################################
    
    # # PARTE 4:
    # # Testando a funcao selu
    # classificador.add(Dense(units = 8, activation='selu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 8, activation='selu',kernel_initializer='random_uniform'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    # # Testando a funcao elu
    # classificador.add(Dense(units = 8, activation='elu',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 8, activation='elu',kernel_initializer='random_uniform'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    # # Testando a funcao softsign
    # classificador.add(Dense(units = 8, activation='softsign',kernel_initializer='random_uniform', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 8, activation='softsign',kernel_initializer='random_uniform'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    #######################################################################################################################
    
    # # PARTE 5:
    # # Mudando o 
    # # Para o RandomNormal 
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_normal', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_normal'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    # # Mudando o 
    # # Para o TruncatedNormal 
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='truncated_normal', input_dim = 30))
    # # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    # classificador.add(Dropout(0.2))
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='truncated_normal'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    
    #######################################################################################################################
    
    # # Adicionando mais uma camada oculta
    # classificador.add(Dense(units = 8, activation='relu',kernel_initializer='random_uniform'))
    # # Zerando mais 20% dos valores da camada oculta
    # classificador.add(Dropout(0.2))
    # # Fazendo a camada de saida
    # classificador.add(Dense(units=1,activation='sigmoid'))
    # # Configurando alguns parametros do otmizador, que no caso é o Adam
    # # lr -> learning rate (Taxa de aprendizagem)
    # # decay -> decremento de lr, ou seja, o quanto o lr vai decaindo
    # # clipvalue -> "prende" o valor, nao deixando-o sair muito do padrao
    # otimazador = keras.optimizers.Adam(lr=0.001,decay=0.001,clipvalue=0.5)
    # # Compilando a rede neural
    # # Optimizer -> funcao de ajuste dos pesos
    # # loss -> funcao de perda(tratar erro)
    # # metrics -> Metrica utilizada para avaliação
    # classificador.compile(optimizer=otimazador,loss='binary_crossentropy', metrics=['binary_accuracy',])
    # # Retorna o cassificador
    
    ###########################################################################################################################
    
    # Juntando os melhores resultados
    classificador = Sequential()
    classificador.add(Dense(units = 32, activation='elu',kernel_initializer='random_uniform', input_dim = 30))
    # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 32, activation='elu',kernel_initializer='random_uniform'))
    # Zerando mais 20% dos valores da camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 32, activation='elu',kernel_initializer='random_uniform'))
    # Zerando mais 20% dos valores da camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 32, activation='elu',kernel_initializer='random_uniform'))
    # Zerando mais 20% dos valores da camada oculta
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 32, activation='elu',kernel_initializer='random_uniform'))
    # Zerando mais 20% dos valores da camada oculta
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units=1,activation='sigmoid'))
    otimazador = keras.optimizers.Adam(lr=0.001,decay=0.001,clipvalue=0.5)
    classificador.compile(optimizer=otimazador,loss='binary_crossentropy', metrics=['binary_accuracy',])
    return classificador

# Criando rede neural
classificador = KerasClassifier(build_fn=criarRede, epochs=100, batch_size=10)
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
media = resultados.mean()
desvio = resultados.std()

################################################################################################################################################
#  Sem mudar nada a media tinha um valor de 0.856
#  PARTE 1:
## Ao se adicioar duas camadas a media foi para um valor de 0.882 -> quase o esperado para a plicação

#  PARTE 2:
## Ao mudar o numero de neuronios para 32 o valor da media foi para 0.872 -> abaixo de adicionar duas camadas a mais
## Mudar o numero de neuronios o valor da media foi para 0.837 -> abaixo de adicionar duas camadas a mais

#  PARTE 3:
## Mudando o valor do Dropout para 0.3 -> media = 0.83

# PARTE 4:
## Mudando a funcao de ativação
## selu -> media = 0.87
## elu -> media = 0.884
## softsign -> media = 0.868

# PARTE 5:
## Mudando a kernel_initializer
## RandomNormal -> media = 0.801
## TruncatedNormal -> media = 0.865



