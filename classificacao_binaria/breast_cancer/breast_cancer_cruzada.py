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
# Importando cross_val_score - utilizado para fazer a divisao da base de dados e fazer a validação cruzada
from sklearn.model_selection import cross_val_score
# Lendo as variaves de entrada e salvando em previsores
previsores = pd.read_csv('entradas_breast.csv')  
# Lendo as variaves de saida e salvando em classe
classe = pd.read_csv('saidas_breast.csv')  
# funcao q cria a rede neural
def criarRede():
    # Criando rede neural
    classificador = Sequential()
    # Adicionando a primeira camada oculta
    # Units - quantos neuronios faram parte da camada oculta
    # Calculo do Units ((30+1)/2) -> ((numero de entradas+n de neuronios na camada de saida)/2) -> arrredondando para mais
    # Activation -> funcao de ativação 
    # kernel_initializer -> Inicializacao dos pesos 
    # input_dim ->  quantidade de elementos q tem na camada de entrada
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform', input_dim = 30))
    # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    classificador.add(Dropout(0.2))
    # Adicionando mais uma camada oculta
    classificador.add(Dense(units = 16, activation = 'relu', 
                        kernel_initializer = 'random_uniform'))
    # Serve para zerar a porcentagem de valores passado dentro dele. Por exemplo: 0.2 -> zera 20% dos valores
    classificador.add(Dropout(0.2))
    # Fazendo a camada de saida
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    # Configurando alguns parametros do otmizador, que no caso é o Adam
    # lr -> learning rate (Taxa de aprendizagem)
    # decay -> decremento de lr, ou seja, o quanto o lr vai decaindo
    # clipvalue -> "prende" o valor, nao deixando-o sair muito do padrao
    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5)
    # Compilando a rede neural
    # Optimizer -> funcao de ajuste dos pesos
    # loss -> funcao de perda(tratar erro)
    # metrics -> Metrica utilizada para avaliação
    classificador.compile(optimizer = otimizador, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    # Retorna o cassificador
    return classificador

# Criando rede neural
# build_fn -> Funcao q cria rede neural
# epochs -> numero de epocas
classificador = KerasClassifier(build_fn = criarRede,
                                epochs = 100,
                                batch_size = 10)
# Fazendo os testes 
# X -> atributos previsores
# cv -> quantidade de vezes q vc quer fazer o teste
# scoring -> como deseja retornar os resultados
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
# Medias dos resultados do cv
media = resultados.mean()
# Fazer calculo de desvio padrao
desvio = resultados.std()




