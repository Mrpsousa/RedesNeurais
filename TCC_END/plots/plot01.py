import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#data inportation
dados = pd.read_csv('train.csv')
#print(data.head())
#print(type(data))

#sns.scatterplot(x='Producao', y='tempA', data=dados) #relating between X e Y
#sns.scatterplot(x='Producao', y='chuvaA', data=dados) #relating between X e Y
#sns.relplot(x='chuvaA', y='Producao', kind='line', data=dados) #relating between X e Y with simple lines
sns.lmplot(x='chuvaA', y='Producao', data=dados)
plt.show()