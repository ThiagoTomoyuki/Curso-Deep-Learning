import numpy as np
# StepFunction - Utilizada somente para problemas linearmente separaveis
# retorna so 0 ou 1
def stepFunction(soma):
    if(soma>=1):
        return 1
    return 0

# SigmoideFunction - Utilizada para problemas de classificação binaria
# valores maximos entre 0 e 1
def sigmoideFunction(soma):
    return (1/(1+np.exp(-soma)))

# TahnFunction - Utilizada para problemas de classificação binaria 
# valores maximos entre -1 e 1
def tahnFunction(soma):
    return ((np.exp(soma)-np.exp(-soma))/(np.exp(soma)+np.exp(-soma)))

teste1 = stepFunction(1)
teste2 = sigmoideFunction(0.358)
teste3 = tahnFunction(-0.358)

# Aula 15

# ReluFunction - Utilizada em redes neurais convulocionais (tambem quando é adicionado muitas camadas a rede neural)
#Retorna valores caso o numero seje negativo retorna 0 caso contrario retorna o proprio numero
def reluFunction(soma):
    if(soma>=0):
        return soma
    return 0

# LinearFunction - Utilizada para regressão
# Retorna o proprio valor passado
def linearFunction(soma):
    return soma

# Funcao Softmax - usado para saber probabilidades 
# Retorna a probabilidade 
def sofmaxFunction(x):
    ex=np.exp(x)
    return(ex/ex.sum())

teste4 = reluFunction(0.45)
teste5 = linearFunction(-0.34)
valores=[5.0, 2.0, 1.3]
print(sofmaxFunction(valores))

# Exercicios testes

P2_sigmoide = sigmoideFunction(2.1)
P2_tahn = tahnFunction(2.1)
P2_relu = reluFunction(2.1)
P2_linear = linearFunction(2.1)


