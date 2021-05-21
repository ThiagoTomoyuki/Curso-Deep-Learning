# Importando pandas
import pandas as pd
# Importando o numpy
import numpy as np
# Importando model_from_json -> para abrir arquivos json
from keras.models import model_from_json
# Abrir arquivo 'classificador_breast.json' e salvar na variavel arquivo
# 'r' -> Operacao de leitura
arquivo = open('classificador_breast.json','r')
# Estrutura da rede / Leitura do arquivo 
estrutura_rede = arquivo.read()
# Fechar arquivo para liberar memoria
arquivo.close()
# Transformar arquivo de json para o formato antigo desejado
classificador = model_from_json(estrutura_rede)
# Leitura dos pesos salvos
classificador.load_weights('classificador_breast.h5')

# Novo registro
novo = np.array([[15.80, 8.34, 118, 900, 0.1, 0.26, 0.08, 0.134, 0.178,
                  0.2, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.3, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])
# Devolve se tem(1) ou n(0) cancer
previsao = classificador.predict(novo)
previsao = (previsao>0.5)

# Lendo as variaves de entrada e salvando em previsores
previsores = pd.read_csv('entradas_breast.csv')  
# Lendo as variaves de saida e salvando em classe
classe = pd.read_csv('saidas_breast.csv')  
# Avaliacao da rede carregado do disco
classificador.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
# Avaliacao do resultado
resultado= classificador.evaluate(previsores,classe)
