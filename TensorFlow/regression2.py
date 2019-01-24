# using simlpe linear regression  (only with  X, Y)
import pandas as pd   
base = pd.read_csv('house-prices.csv')
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler
import numpy as np 

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
xph = tf.placeholder(tf.float32, [21613, 1]) # tf.float32 type of date, [a,b]   