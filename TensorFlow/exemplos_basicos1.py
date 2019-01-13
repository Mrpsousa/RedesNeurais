"""
Created on Sun Jan  6 16:33:11 2019

@author: roger
"""
import tensorflow as tf
#import numpy as np

valor1 = tf.constant(2)
valor2 = tf.constant(3)

print(type(valor1))
print(valor1)

soma1 = valor1 + valor2
#aqui só foi montado o " grafo" da fórmula

type(soma1)
print(soma1)
#até o momento não temos execução dos cods

#para a exuceção de cod, usamos uma "Session" 
with tf.Session() as s:
    a = s.run(soma1)
print("Valor da Soma: ", a)

text1 = tf.constant('texto 1')
text2 = tf.constant('texto 2')
print(type(text1))
print(text1)

#usando variávais
print("--------------------------------------")

valor = tf.constant(15, name = 'valor')
soma = tf.Variable(valor + 5, name = 'soma')

init = tf.initialize_all_variables() #inicializar as variáveis
with tf.Session() as s:
    s.run(init)
    s = s.run(soma)
print("Soma2: ", s)


#usando vetor
print("--------------------------------------")

vetor = tf.constant([1, 5, 10], name = 'vetor')
soma3 = tf.Variable(vetor + 5, name = "soma3")
print(vetor)
init2 = tf.initialize_all_variables() #inicializar as variáveis
with tf.Session() as s:
    s.run(init2)
    print("Soma3: ", s.run(soma3))

#'for'
print("--------------------------------------")

val = tf.Variable(0, name = "val")
init3 = tf.initialize_all_variables()
with tf.Session() as s:
    s.run(init3)
    for i in range(5):
        val = val + 1
        print(s.run(val))


#soma vetor
print("--------------------------------------")

vetA = tf.constant([1,1,1], name = "vetorA")
vetB = tf.constant([2,2,2], name = "vetorB")
soma4 = tf.Variable(vetA + vetB, name = "soma4")

init4 = tf.initialize_all_variables()
with tf.Session() as s:
    s.run(init4)
    print("Soma de vetores: ", s.run(soma4))

#soma matriz
print("--------------------------------------")

matzA = tf.constant([[1,1,1], [2,2,2]], name = "matrizA")
matzB = tf.constant([[3,3,3], [1,1,1]], name = "matrizB")

#pode ser feito asim
soma5 = tf.Variable(matzA + matzB, name = "soma5")

init5 = tf.initialize_all_variables()
with tf.Session() as s:
    s.run(init5)
    print("Soma de Matrizes: ", s.run(soma5))

 #ou assim
soma6 = tf.add(matzA, matzB)
with tf.Session() as s:
    print("\n Soma de Matrizes2: ", s.run(soma6))


print("---------------------------------------")
matzA2 = tf.constant([[2], [3]], name = "matrizA2")
matzB2 = tf.constant([[3,3,3], [1,1,1]], name = "matrizB2")

soma7 = tf.add(matzA2, matzB2)
with tf.Session() as s:
    print("\n Soma de Matrizes3: ", s.run(soma7))

#multiplicação matriz
print("--------------------------------------")

matzA2 = tf.constant([[1,2], [3,4]])
matzB2 = tf.constant([[-1,3], [4,2]])
multMatz = tf.matmul(matzA2, matzB2)

with tf.Session() as s:
    print("\n Multi Matrizes3: ", s.run(multMatz))

#produto escalar (dot product)
print("--------------------------------------")


matzA3 = tf.constant([[-1.0, 7.0, 5.0]])
matzB3 = tf.constant([[0.8, 0.1, 0.0]])
multiplicaMatz = tf.mul(matzA2, matzB2) #multiplico linearmente
somaD = tf.reduce_sum(multiplicaMatz) # soma o resultado das multiplicações

with tf.Session() as s:
    print("valor do Dot Product: ", s.run(somaD))

#placeholders - um tipo de "variável" aonde se pode atribuir dados
#os dados só são atribuidos "mais tarde"
print("\nplaceholders\n")
print("--------------------------------------")
p1 = tf.placeholder('float', None)#'float' = tipo do dado que recebe, none = sem dimensão
operacao = p1 + 2

with tf.Session() as s:
    resul = s.run(operacao, feed_dict = {p1: [1,2,3]}) #feed_dict passa um dicionário com os paramentros de alimentação do placeholder
    print("Resultado (placeholder1)", resul)


p2 = tf.placeholder('float', [None, 3])#'float' = tipo do dado que recebe, Matrix, none = não sei quantas linhas, 3 colunas
operacao2 = p2 * 3

with tf.Session() as s:
    dados = [[1,2,3], [1,2,3]]
    resul2 = s.run(operacao2, feed_dict = {p2: dados}) #feed_dict passa um dicionário com os paramentros de alimentação do placeholder
    print("Resultado (placeholder2)", resul2)

print("\n Dashbord e Grafos\n")
print("--------------------------------------")

#multiplicação simples = tf.mul
#multiplicação matriz  = tf.matmul
a = tf.add(2, 2, name = "add")
b = tf.mul(a, 3, name = "mult11")
c = tf.mul(b, a, name = "mult22")

with tf.Session() as s:
    writer = tf.summary.FileWriter('output', s.graph)
    print(s.run(c))
    writer.close()