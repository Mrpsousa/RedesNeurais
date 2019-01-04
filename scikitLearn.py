from sklearn.neural_network import MLPClassifier #MLP = multi layer perceptron
from sklearn import datasets

iris = datasets.load_iris()

entradas = iris.data #carrega as entradas do dataset
saidas = iris.target #carrega as saidas do dataset

redeNeural = MLPClassifier(verbose=True, max_iter=1000) # criando a rede, loss = erro, max_i = iteracoes
redeNeural.fit(entradas, saidas) # fazendo o treinamento | fit "encaixar entradas nas saidas"(tem essa ideia)

#scikit-learn.org - documentation