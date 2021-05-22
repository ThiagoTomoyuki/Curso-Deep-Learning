import pandas as pd
from keras.layers import Dense, Dropout , Activation, Input
from keras.models import Model

# Carregando base de dados para a variavel base
base = pd.read_csv('games.csv')

# Apagando campos 'desnecessarios' para essa aplicação

# Apagando a coluna do Other_Sales
base = base.drop('Other_Sales', axis=1)
# Apagando a coluna do Other_Sales
base = base.drop('Global_Sales', axis=1)
# Apagando a coluna do Other_Sales
base = base.drop('Developer', axis=1)

# Pre-Processamento -> prepara os dados antes da aplicacao

# Apagando dados nulos
# Obs: axis = 0 -> apaga a linha da tabela
# Obs: axis = 1 -> apaga a coluna da tabela
base = base.dropna(axis = 0)
# Correcao de valores inconsistentes
# Removendo todos do NA_Sales que sao menores que 1
base = base.loc[base['NA_Sales'] > 1]
# Removendo todos do EU_Sales que sao menores que 1
base = base.loc[base['EU_Sales'] > 1]

# Analisando quantidades de nomes dos Jogos na base de dados
base['Name'].value_counts()
# Guardando na variavel nome_jogos o nome dos jogos
nome_jogos = base.Name
# Apagando a coluna dos nomes dos jogos
base = base.drop('Name', axis=1)

# Separando atributos

# Separando os previsores da tabela
previsores = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
# Separando as vendas reais da tabela
venda_na = base.iloc[:, 4].values
venda_eu = base.iloc[:, 5].values
venda_jp = base.iloc[:, 6].values

# Transformando valores categoricos em numericos

# Importando o LabelEncoder, o OneHotEncoder e o ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])

# Mudando a segunda coluna para o seguinte formato (binario)
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# Criando uma rede neural

camada_entrada = Input(shape=(61))
camada_oculta1 = Dense(units = 32, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32, activation='sigmoid')(camada_oculta1)
camada_saida_na = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida_eu = Dense(units = 1, activation='linear')(camada_oculta2)
camada_saida_jp = Dense(units = 1, activation='linear')(camada_oculta2)

regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida_na,camada_saida_eu,camada_saida_jp])
regressor.compile(optimizer = 'adam',
                  loss='mse')
regressor.fit(previsores,[venda_na,venda_eu,venda_jp], epochs=5000, batch_size=100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(previsores)










