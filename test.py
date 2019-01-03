import numpy as np

def sigmoid(input):
    return 1/(1 + np.exp(-input))

def sigmoidDerivada(sig): 
    return sig * (1 - sig)

entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]]) 

saidas = np.array([[0],[1],[1],[0]]) 

pesos0 = 2 * np.random.random((2,3)) -1 
pesos1 = 2 * np.random.random((3,1)) -1 

epocas = 500
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


