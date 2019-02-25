# linear regression  multiple with stimator
import pandas as pd 
base = pd.read_csv('house-prices.csv')
#print(base.head())
#print(base.columns)
'''here i got all columns names, and make a list with it '''
used_column = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']

#print(used_column)
#now, call the csv again, with "used_column" with parameter and the csv will be load just with the columns wich i want
base = pd.read_csv('house-prices.csv', usecols = used_column) #usecols used to select the columns that u want
#normalizing the datas - here i will put data on scale between 0 and 1
#P.S. in padronization (standScaler), the values is between positive and negative
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
base[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[[ 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])

scaler_y = MinMaxScaler()
base[['price']] = scaler_x.fit_transform(base[['price']])


#put as padronization values on new variables
x = base.drop('price', axis = 1) #getting all columns without price column
y = base[['price']]
#print(base.head)
 
#print(x.head)
#next, do the "column feature numeric",  transforming the data into formats that Estimators can use

prospective_columns = used_column[1:17] #without 'price' column that is the zero column 
#print(prospective_columns)

import tensorflow as tf 

column = [tf.feature_column.numeric_column(key = i) for i in prospective_columns] 
#print(column[0])


#here, we go use 70% of data base to traning and 30% for testing
#testing/training 
from sklearn.model_selection import train_test_split
x_training, x_test, y_training, y_test = train_test_split(x,y, test_size = 0.3)

#print(x_training.shape)
#print(y_training.shape) #print = (15129, 1)
#print(y_test)

#training function
training_function = tf.estimator.inputs.pandas_input_fn(x = x_training, y = y_training, batch_size = 32, num_epochs = None, shuffle = True)

#testing function
testing_function = tf.estimator.inputs.pandas_input_fn(x = x_test, y = y_test, batch_size = 32, num_epochs = 10000, shuffle = False)

#regression function
regression_function = tf.estimator.LinearRegressor(feature_columns = column)
print("até aqui, ta tudo bem, aparece apenas um warning: WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmp37dg_6wy")

#effective training 
regression_function.train(input_fn = training_function, steps = 10000)