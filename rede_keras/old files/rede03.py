from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from matplotlib import pyplot as plt
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor



def get_data():
    #get train data
    train_data_path ='train-cacau.csv'
    train = pd.read_csv(train_data_path)
    uniques = train[0:1]

    #get test data
    test_data_path ='test-cacau.csv'
    test = pd.read_csv(test_data_path)
    #uniques = pd.factorize(['b', 'b', 'a', 'c', 'b'])
    
    return train , test


def get_combined_data():
  #reading train data
  train , test = get_data()

  target = train.Producao
  train.drop(['Producao'],axis = 1 , inplace = True)

  combined = train.append(test)
  combined.reset_index(inplace=True)
  #combined.drop(['index', 'Id'], inplace=True, axis=1)
  return combined, target
print("AQUI")
print(target)
