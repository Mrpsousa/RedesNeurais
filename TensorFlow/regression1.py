import numpy as np
import tensorflow as tf 

X = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])
Y = np.array([[871], [1132], [1042], [1356], [1488], [1638], [1569], [1754], [1866], [1900]])

#doing escaling of values
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler() #creating a object of type "StandardScaler"
X = scaler_x.fit_transform(X)
print(X)