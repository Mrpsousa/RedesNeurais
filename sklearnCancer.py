import numpy as np
from sklearn import datasets


def sigmoid(input):
    return 1/(1 + np.exp(-input))

def sigmoidDerivada(sig): 
    return sig * (1 - sig)

base = datasets.load_breast_cancer()
entradas = base.data
valoresSaidas =  base.target #aqui ta 1 linha e varias colunas, preciso inverter
saidas = np.empty([569,1], dtype = int) #569 linhas por 1 coluna (temos 569 registros), vazio
for i in range(569): #invertendo
     saidas[i] = valoresSaidas[i]


pesos0 = 2 * np.random.random((30,3)) -1 #como eu tenho 30 atributos de entrada, preciso de 30 pesos 
                                         # e temos 3 neuronios na camada escondida, dae fica 30, 3, temos 1 camada oculta apenas com 3 neuronios
pesos1 = 2 * np.random.random((3,1)) -1  #(3,1) 3 pesos da camada escondida e finaliza com 1 neuronio na camada saida


epocas = 50000
taxaAprendizagem = 0.6
momento = 1

for j in range(epocas):
     camadaEntrada = entradas
     somaSinapse0 = np.dot(camadaEntrada, pesos0)
     camadaOculta = sigmoid(somaSinapse0)

     somaSinapse1 = np.dot(camadaOculta, pesos1)
     camadaSaida = sigmoid(somaSinapse1)

     erroCamadaSaida = saidas - camadaSaida 
     mediaAbs = np.mean(np.abs(erroCamadaSaida))
     
     print("Erro: " + str(mediaAbs))

     derivadaSaida = sigmoidDerivada(camadaSaida)
     deltaSaida = erroCamadaSaida * derivadaSaida 
     pesos1Transposta = pesos1.T

     deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
     deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    

     camadaOcultaTransposta = camadaOculta.T
     pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida) 
     pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

     camadaDeEntradaTransposta = camadaEntrada.T
     pesosNovo0 = camadaDeEntradaTransposta.dot(deltaCamadaOculta)
     pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)


#temos 37% de erro, logo a rede tem 62% de acerto(fora os "quebrados")