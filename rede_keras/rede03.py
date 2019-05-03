import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

dataset = pd.read_csv('test-cacau.csv')
print(dataset.head())