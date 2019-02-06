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
#print(base.head)
#normalizing the price 
scaler_y = MinMaxScaler()
base[['prince']] = scaler_y.fit_transform(base[['prince']])