# using simlpe linear regression  (only with  X, Y)
import pandas as pd   
base = pd.read_csv('house-prices.csv')
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler

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

print(x)