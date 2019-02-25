#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
base = pd.read_csv('house-prices.csv')
base.head()


# In[51]:


base.columns


# In[52]:


colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']


# In[53]:


colunas_usadas


# In[54]:


base = pd.read_csv('house-prices.csv', usecols = colunas_usadas)


# In[55]:


base.head()


# In[56]:


from sklearn.preprocessing import MinMaxScaler
scaler_x = MinMaxScaler()
base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']] = scaler_x.fit_transform(base[['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long']])


# In[57]:


base.head()


# In[58]:


scaler_y = MinMaxScaler()
base[['price']] = scaler_y.fit_transform(base[['price']])


# In[59]:


base.head()


# In[60]:


X = base.drop('price', axis = 1)
y = base.price


# In[61]:


X.head()


# In[62]:


type(X)


# In[63]:


y.head()


# In[64]:


type(y)


# In[65]:


previsores_colunas = colunas_usadas[1:17]
previsores_colunas


# In[66]:


import tensorflow as tf


# In[67]:


colunas = [tf.feature_column.numeric_column(key = c) for c in previsores_colunas]


# In[68]:


print(colunas[10])


# In[69]:


from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size = 0.3)


# In[70]:


X_treinamento.shape


# In[71]:


y_treinamento.shape


# In[72]:


X_teste.shape


# In[73]:


funcao_treinamento = tf.estimator.inputs.pandas_input_fn(x = X_treinamento, y = y_treinamento,
                                                        batch_size = 32, num_epochs = None, shuffle = True)


# In[74]:


funcao_teste = tf.estimator.inputs.pandas_input_fn(x = X_teste, y = y_teste,
                                                   batch_size = 32, num_epochs = 10000, shuffle = False)


# In[75]:


regressor = tf.estimator.LinearRegressor(feature_columns=colunas)


# In[76]:


regressor.train(input_fn=funcao_treinamento, steps = 10000)


# In[77]:


metricas_treinamento = regressor.evaluate(input_fn=funcao_treinamento, steps = 10000)


# In[78]:


metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps = 10000)


# In[79]:


metricas_treinamento


# In[80]:


metricas_teste


# In[81]:


funcao_previsao = tf.estimator.inputs.pandas_input_fn(x = X_teste, shuffle = False)


# In[91]:


previsoes = regressor.predict(input_fn=funcao_previsao)


# In[92]:


list(previsoes)


# In[93]:


valores_previsoes = []
for p in regressor.predict(input_fn=funcao_previsao):
    valores_previsoes.append(p['predictions'])


# In[94]:


valores_previsoes


# In[95]:


import numpy as np
valores_previsoes = np.asarray(valores_previsoes).reshape(-1,1)


# In[96]:


valores_previsoes


# In[97]:


valores_previsoes = scaler_y.inverse_transform(valores_previsoes)
valores_previsoes


# In[98]:


y_teste


# In[99]:


y_teste.shape


# In[100]:


y_teste2 = y_teste.values.reshape(-1,1)
y_teste2.shape


# In[101]:


y_teste2 = scaler_y.inverse_transform(y_teste2)


# In[102]:


y_teste2


# In[103]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, valores_previsoes)


# In[104]:


mae

