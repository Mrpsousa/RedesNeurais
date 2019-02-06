# using simlpe linear regression  (only with  X, Y)
# network to predict the price of the houses
import pandas as pd   
base = pd.read_csv('house-prices.csv')
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
import numpy as np 
import matplotlib.pyplot as plt 

#print(base.head())
#print(base.count())
#print(base.shape())

#separete the variables that we will use (take dates of two columns to do the linear regression)
x = base.iloc[:,5].values.astype(float) # iloc return columns and rows that i want, values - translate to numpay array
#reshape, include new rows and columns(-1, not want change lines, 1 want 1 new column)
x = x.reshape(-1,1)
# X = square meters of house, Y = value of house
#print(X)
y = base.iloc[:,2].values.astype(float)
y = y.reshape(-1,1)
#print(Y)
#now, doing the scaling of X and Y


scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)

scaler_y = StandardScaler()
y = scaler_x.fit_transform(y)

#print(x)
#import matplotlib.pylab as plt  
#plt.scatter(x, y)
 
#simple linear regression formula
#y = b0 + b1 * x
np.random.seed(0)
#print(np.random.rand(2)) # to generate 2 random numbers
np.random.rand(2)
#this "random.rand" did generate two values, ([0.417022],[0.72032449])
#creating variables
b0 = tf.Variable(0.41)
b1 = tf.Variable(0.72)

#creating a placeholder, to receiver data of variables Y and X
#the placeholders will receiver data in  processing time
batch_size = 32 # the registers will go in parts of 32
xph = tf.placeholder(tf.float32, [batch_size, 1]) # tf.float32 type of date, [a,b] tamanho do registro
yph = tf.placeholder(tf.float32, [batch_size, 1]) # [batch_size, b] batch_size = rows, b columns in this case

#creation of model that will representate our predictions 
y_model = b0 + b1 * xph # feeding the formula with data pieces of 32 
error = tf.losses.mean_squared_error(yph, y_model)# compare the answer of y_model with correct values of y
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001) # training using gradient descent
training = optimizer.minimize(error) # to do a minimizer of variable "error", minimizing the error
init = tf.global_variables_initializer()


#begin the execution
with tf.Session() as sess: #here is the process of training
    sess.run(init)
    for i in range(10000):
        indices = np.random.randint(len(x), size = batch_size)#here, need to make a draw of the indices(indices in range every 32)
        feed = {xph: x[indices], yph: y[indices]}#now, filling in the placeholders, every 32("of 32 in 32"), here in indices, we get a list of 32 indices
        #in xph = values in square meters | yph = valures os houses
        sess.run(training, feed_dict = feed)#here, the true trainning 
    b0_final, b1_final = sess.run([b0, b1]) # put final values on others variables, the values updated 

    '''print("Final Values \n")
       print("b0 = ", b0_final)
       print("\nb1 = ", b1_final)'''
    

#print(np.random.randint(len(x), size = batch_size)) #to look at the 32 registers that were selected

previsions = b0_final + b1_final * x # doing the previsions of houses, with base in values of meters

#print("Previões dos valores com base na 'metragem' ", previsions)

''' need to be fixed
plt.plot(y, x, 'o', previsions, color = 'red') 
plt.show() '''

#desescaling Y valuers
y1 = scaler_y.inverse_transform(y)
#desescaling ... 
previsions1 = scaler_y.inverse_transform(previsions)

#now y and previsions have te real valuers

#doing the calcule of mean absolute error
#pass the answers that i know (Y) and the previsions

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y1, previsions1)
#the 'mae' is the value of erros, or for more or for less
print("\n erro", mae)
