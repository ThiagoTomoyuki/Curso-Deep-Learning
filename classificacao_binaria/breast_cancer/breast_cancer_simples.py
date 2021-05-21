# Importando pandas
import pandas as pd
# Lendo as variaves de entrada e salvando em previsores
previsores = pd.read_csv('entradas_breast.csv')  
# Lendo as variaves de saida e salvando em classe
classe = pd.read_csv('saidas_breast.csv')   

# Importando uma funcao que faz a divisao da base de dados de treinamento de teste
from sklearn.model_selection import train_test_split
# Fazendo a divisao de 4 variaveis 2 para testes e duas para treinamentos
previsores_treinamento,previsores_testes,classe_treinamento,classe_test = train_test_split(previsores, classe, test_size=0.25)
# Importando keras
import keras
# Classe utilizada para criação da rede neural
from keras.models import Sequential
# Importando camadas densas(cada neuronio é ligado a todos os demais na camada oculta) para redes neurais
from keras.layers import Dense
# Criando rede neural
classificador = Sequential()
# Adicionando a primeira camada oculta
# Units - quantos neuronios faram parte da camada oculta
# Calculo do Units ((30+1)/2) -> ((numero de entradas+n de neuronios na camada de saida)/2) -> arrredondando para mais
# Activation -> funcao de ativação 
# kernel_initializer -> Inicializacao dos pesos 
# input_dim ->  quantidade de elementos q tem na camada de entrada
classificador.add(Dense(units = 16, activation='relu',kernel_initializer='random_uniform', input_dim = 30))
# Adicionando mais uma camada oculta
classificador.add(Dense(units = 16, activation='relu',kernel_initializer='random_uniform'))
# Fazendo a camada de saida
classificador.add(Dense(units=1,activation='sigmoid'))

# Configurando alguns parametros do otmizador, que no caso é o Adam
# lr -> learning rate (Taxa de aprendizagem)
# decay -> decremento de lr, ou seja, o quanto o lr vai decaindo
# clipvalue -> "prende" o valor, nao deixando-o sair muito do padrao
otimazador = keras.optimizers.Adam(lr=0.001,decay=0.001,clipvalue=0.5)

# Compilando a rede neural
# Optimizer -> funcao de ajuste dos pesos
# loss -> funcao de perda(tratar erro)
# metrics -> Metrica utilizada para avaliação
classificador.compile(optimizer=otimazador,loss='binary_crossentropy', metrics=['binary_accuracy',])



# Fazer o treinamento
# batch_size ->  calcula o erro para, neste caso, 10 registros depois atualiza os pesos
# epochs ->  quantidade de vezes treinar a rede(fazer os ajustes dos pesos)
classificador.fit(previsores_treinamento,classe_treinamento,batch_size=10,epochs=100)

# Salvar na variavel pesos0 os valores dos pesos da primeira camada oculta
pesos0 = classificador.layers[0].get_weights()
# Printando os valores dos pesos0
print(pesos0)
# Salvar na variavel pesos1 os valores dos pesos da segunda camada oculta
pesos1 = classificador.layers[1].get_weights()
# Salvar na variavel pesos2 os valores da camada de saida 
pesos2 = classificador.layers[2].get_weights()

# Fazer avaliacao correta usando a base de testes

# Fazendo da maneira do SKLEARNING

# Retorna um valor de probabilidade
previsoes = classificador.predict(previsores_testes)
# Mudando a tabela previsoes para tabela de bool para facilitar a visualizacao
previsoes = (previsoes>0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
# Comparar dois vetores passados
precisao = accuracy_score(classe_test,previsoes)
# Matriz de erros e acertos da rede neural
matriz = confusion_matrix(classe_test,previsoes)

# Fazendo da maneira utilizando o KERAS

# Faz os calculos e a avaliacao direto guardado na variavel resultado
resultado = classificador.evaluate(previsores_testes, classe_test)