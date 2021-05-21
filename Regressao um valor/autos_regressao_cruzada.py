import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasRegressor

# Carregando base de dados
base = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

# Preprocessamento da base de dados

# Limpando alguns atributos da tabela (que n ajudam na previsao do preco) 
# Apagando a coluna da dateCrawled
base = base.drop('dateCrawled', axis = 1)
# Apagando a coluna da dateCreated
base = base.drop('dateCreated', axis = 1)
# Apagando a coluna da nrOfPictures
base = base.drop('nrOfPictures', axis = 1)
# Apagando a coluna da postalCode
base = base.drop('postalCode', axis = 1)
# Apagando a coluna da lastSeen
base = base.drop('lastSeen', axis = 1)

# Analisando quantidades de nomes dos veiculos da base de dados
base['name'].value_counts()
# Apagando a coluna da name, pois outros atributos suprem a nescecidade do nome do carro
base = base.drop('name', axis = 1)
# Analisando quantidades de veiculos comprado privados ou n
base['seller'].value_counts()
# Apagando a coluna da seller
base = base.drop('seller', axis = 1)
# Analisando quantidades carros vendidos no leilao (muita variabilidade)
base['offerType'].value_counts()
# Apagando a coluna da offerType
base = base.drop('offerType', axis = 1)

# Correcao de valores inconsistentes

# Analisando a coluna price
i1 = base.loc[base.price <= 10]
i2 = base.loc[base.price > 350000]
# Jogando para a base de dados apenas os valores que os precos sao maiores que 10
base = base[base.price > 10]
# Jogando para a base de dados apenas os valores que os precos sao menores que 350000
base = base[base.price < 350000]

# Corrigindo alguns valores nulos na base de dados

# Analisando a coluna do tipo do veiculo
base.loc[pd.isnull(base['vehicleType'])]
# Observando qual o tipo do veiculo que mais aparece na base de dados
base['vehicleType'].value_counts() #limousine
# Analisando a gearbox (para ver se é manual ou automatico)
base.loc[pd.isnull(base['gearbox'])]
# Observando qual o gearbox que mais aparece
base['gearbox'].value_counts() #manuell
# Analisando o modelo do carro
base.loc[pd.isnull(base['model'])]
# Observando qual o modelo que mais aparece
base['model'].value_counts() #golf
# Analisando qual combustivel dos carros
base.loc[pd.isnull(base['fuelType'])]
# Observando qual o tipo de combustivel mais usado
base['fuelType'].value_counts() #benzin = gasolina
# Analisando se os carros ja foram consertados 
base.loc[pd.isnull(base['notRepairedDamage'])]
# Observando se os carros ja foram consertados 
base['notRepairedDamage'].value_counts() #nein

# Subistituindo valores nulos pelos mais usados

# Criando um Json com os valores que dejamos substituir
valores = {'vehicleType':'limousine',
           'gearbox':'manuell',
           'model':'golf',
           'fuelType':'benzin',
           'notRepairedDamage':'nein'}
# Subistituindo os valores
base = base.fillna(value = valores)

# Separando atributos

# Separando os previsores da tabela
previsores = base.iloc[:, 1:13].values
# Separando os precos reais da tabela
preco_real = base.iloc[:, 0].values

# Transformando valores categoricos em numericos

# Mudando a segunda coluna para o seguinte formato (binario)
# 0 = 0 0 0 
# 2 = 0 1 0 
# 3 = 0 0 1 

labelencoder_previsores = LabelEncoder()
 
onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

# Criando uma funcao que cria a rede neural

def criar_rede(loss):   
    regressor = Sequential()    
    regressor.add(Dense(units = 158, activation = 'relu', input_dim = 316))
    regressor.add(Dense(units = 158, activation = 'relu'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(loss = loss, optimizer = 'adam',
                      metrics = ['mean_absolute_error'])
    return regressor

# Importando o GridSearchCV 

from sklearn.model_selection import GridSearchCV

regressor = KerasRegressor(build_fn = criar_rede,
                                      epochs=100,
                                      batch_size=300)
# Parametros 
parametros = {'loss': ['neg_mean_absolute_error',
                       'squared_hinge',
                       'mean_squared_logarithmic_error',
                       'mean_squared_error']}
# Verificar o melhor parametro
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parametros,                           
                           cv = 10)
grid_search = grid_search.fit(previsores, preco_real)
melhores_parametros = grid_search.best_params_
'''
resultados = cross_val_score(estimator = regressor,
                            X = previsores,
                            y=preco_real,
                            cv=10,
                            scoring=)
media = resultados.mean()
desvio = resultados.std()
'''
















