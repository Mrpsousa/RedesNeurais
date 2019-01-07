"""
Created on Sun Jan  6 16:33:11 2019

@author: roger
"""
import tensorflow as tf

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
with tf.Session() as sess:
    a = sess.run(soma1)
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
with tf.Session() as sess:
    sess.run(init)
    s = sess.run(soma)
print("Soma2: ", s)


#usando vetor
print("--------------------------------------")

vetor = tf.constant([1, 5, 10], name = 'vetor')
soma3 = tf.Variable(vetor + 5, name = "soma3")
print(vetor)
init2 = tf.initialize_all_variables() #inicializar as variáveis
with tf.Session() as ss:
    ss.run(init2)
    print("Soma3: ", ss.run(soma3))

#'for'
print("--------------------------------------")

val = tf.Variable(0, name = "val")
init3 = tf.initialize_all_variables()
with tf.Session() as sss:
    sss.run(init3)
    for i in range(5):
        val = val + 1
        print(sss.run(val))
