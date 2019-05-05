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


#Load train and test data into pandas DataFrames
train_data, test_data = get_data()

#Combine train and test data to process them together
combined, target = get_combined_data()

#print(combined.describe())



def get_cols_with_no_nans(df,col_type):
    '''
    Arguments :
    df : The dataframe to process
    col_type : 
          num : to only get numerical columns with no nans
          no_num : to only get nun-numerical columns with no nans
          all : to get any columns with no nans    
    '''
    if (col_type == 'num'):
        predictors = df.select_dtypes(exclude=['object'])
    elif (col_type == 'no_num'):
        predictors = df.select_dtypes(include=['object'])
    elif (col_type == 'all'):
        predictors = df
    else :
        print('Error : choose a type (num, no_num, all)')
        return 0
    cols_with_no_nans = []
    for col in predictors.columns:
        if not df[col].isnull().any():
            cols_with_no_nans.append(col)
    return cols_with_no_nans


num_cols = get_cols_with_no_nans(combined , 'num')
cat_cols = get_cols_with_no_nans(combined , 'no_num')

print ('Number of numerical columns with no nan values :',len(num_cols))
print ('Number of nun-numerical columns with no nan values :',len(cat_cols))

'''combined = combined[num_cols + cat_cols]
combined.hist(figsize = (12,10))
plt.show()'''
#----------- "Esse"
'''
train_data = train_data[num_cols + cat_cols]
train_data['Target'] = target

C_mat = train_data.corr()
fig = plt.figure(figsize = (15,15))

sb.heatmap(C_mat, vmax = .8, square = True)
plt.show()'''

#Now, split back combined dataFrame to training data and test data

def split_combined():
    global combined
    train = combined[:9]
    test = combined[9:]

    return train , test 
  
train, test = split_combined()
'''-----> Make the Deep Neural Network
Define a sequential model
Add some dense layers
Use ‘relu’ as the activation function for the hidden layers
Use a ‘normal’ initializer as the kernal_intializer '''

'''Initializers define the way to set the initial random weights of Keras layers.
We will use mean_absolute_error as a loss function
Define the output layer with only one node
Use ‘linear ’as the activation function for the output layer'''

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(20, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(40, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(40, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(40, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


#Define a checkpoint callback
checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]

#Train the model
NN_model.fit(train, target, epochs=350, batch_size=1, validation_split = 0.2, callbacks=callbacks_list)


# Load wights file of the best model :
wights_file = 'Weights_14.6.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'Id_predicao':pd.read_csv('test-cacau.csv').Id,'Producao':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')

predictions = NN_model.predict(test)
make_submission(predictions[:,0],'Prediction result.csv')