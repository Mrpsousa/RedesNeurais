import numpy as np
#rede de aprendizagem do XOR
def sigmoid(input):
    return 1/(1 + np.exp(-input))

def sigmoidDerivada(sig): # vai receber o retorno da funcao sigmoid (funcao Derivada)
    return sig * (1 - sig)

entradas = np.array([[0,0],
                    [0,1],
                    [1,0],
                    [1,1]]) # "xor" 0 xor 0, 0 xor 1 ...

saidas = np.array([[0],[1],[1],[0]]) #saida do xor 

pesos0 = np.array([[-0.424, -0.740, -0.961],
                    [0.358, -0.577, -0.469]])#3 pesos pra casa entrada, no caso sao 2 entradas

pesos1 = np.array([[-0.017], [-0.893], [0.148]]) #3 pesos da camada oculta pra camada de saida

epocas = 100000 #quantas vezes vou executar (um loop de quantidade "epocas")
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
     camadaEntrada = entradas
     somaSinapse0 = np.dot(camadaEntrada, pesos0)# coloca na somaSina.. a multiplicacao e soma (produto linear)
     camadaOculta = sigmoid(somaSinapse0)

     somaSinapse1 = np.dot(camadaOculta, pesos1)# coloca na somaSina.. a multiplicacao e soma (produto linear)
     camadaSaida = sigmoid(somaSinapse1)

     erroCamadaSaida = saidas - camadaSaida #"saidas" sao as saidas esperadas, "camadaSaida" eh a calculada com sigmoide 
     mediaAbs = np.mean(np.abs(erroCamadaSaida))
     #para saber se o erro esta diminuindo (mostra o percentual de erro)
     print("Erro: " + str(mediaAbs))

     derivadaSaida = sigmoidDerivada(camadaSaida)
     deltaSaida = erroCamadaSaida * derivadaSaida # delta = valor pra ajudar a calcular o gradiente

     # >>>>> Formula pro Delta da camada escodida
     #DeltaEscondida = DerivadaSigmoide * peso * Deltasaida
     #pesos1 tem 3 linhas e 1 coluna, logo nao da pra fazer produto linear com "deltaSaida" que tem 1 col e 4 linhas
     #fazer matriz transposta em "pesos1" para que este fique com 3 colunas
     pesos1Transposta = pesos1.T
     #multiplico coluna por coluna de "pesos1" por todas as linhas da coluna de deltaSaida
     deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
     deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
     #>>>> recalculando os novos pesos (Backpropagation)
     #peso(n+1) = (peso(n)'pesoatual' * momento(defindo mas ainda nao explicado)) + (entrada * delta * taxaDeAprendizagem)
    
     #fazer transposta de CamadaOculta para ser possivel o prod linear
     camadaOcultaTransposta = camadaOculta.T
     #calculando valor auxiliar para calculo do peso novo (por Back propagation)
     pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida) 
     pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

     camadaDeEntradaTransposta = camadaEntrada.T
     pesosNovo0 = camadaDeEntradaTransposta.dot(deltaCamadaOculta)
     pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)

#print("---------------------")
#print(entradas)
#print("---------------------")
#print(camadaDeEntradaTransposta)
#print("---------------------")
#print(camadaOculta)
#print("---------------------")
#print(camadaOcultaTransposta)
#print("---------------------")
#print(camadaSaida)
print("---------------------")
print(pesosNovo1)
print("---------------------")
print(pesosNovo0)