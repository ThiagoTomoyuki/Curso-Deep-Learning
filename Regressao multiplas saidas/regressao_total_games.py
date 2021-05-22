import pandas as pd
from keras.layers import Dense, Dropout , Activation, Input
from keras.models import Model

# Carregando base de dados para a variavel base
base = pd.read_csv('games.csv')

base = base.drop('NA_Sales', axis=1)
base = base.drop('EU_Sales', axis=1)
base = base.drop('JP_Sales', axis=1)
base = base.drop('Developer', axis=1)
base = base.drop('Other_Sales', axis=1)
base = base.dropna(axis = 0)


# Analisando quantidades de nomes dos Jogos na base de dados
base['Name'].value_counts()
# Guardando na variavel nome_jogos o nome dos jogos
nome_jogos = base.Name
# Apagando a coluna dos nomes dos jogos
base = base.drop('Name', axis=1)
# Separando os previsores da tabela
previsores = base.iloc[:, [0,1,2,3,5,6,7,8,9]].values
# Separando as vendas reais da tabela
venda_total = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,2,3,8])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

camada_entrada = Input(shape=(303,))
camada_oculta1 = Dense(units = 152, activation='sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 152, activation='sigmoid')(camada_oculta1)
camada_saida_global = Dense(units = 1, activation='linear')(camada_oculta2)


regressor = Model(inputs = camada_entrada,
                  outputs = camada_saida_global)
regressor.compile(optimizer = 'adam',
                  loss='mse')
regressor.fit(previsores, venda_total , epochs=5000, batch_size=100)
