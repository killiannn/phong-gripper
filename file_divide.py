import pandas as pd
import numpy as np 
from numpy.random import RandomState

# Import necessary modules
from sklearn.model_selection import train_test_split


df = pd.read_csv('C:\Pro\ANN\phong_gripper.csv', header= 0, index_col=0)  # type: ignore
df.astype('float')
rng = RandomState()

model_data = df.sample(frac=0.8, random_state=rng)
valid = df.loc[~df.index.isin(model_data.index)]

x_train,x_test,y_train,y_test = train_test_split(model_data.iloc[:,:-4], 
model_data.iloc[:,-4],test_size = 0.25)

x_valid = valid.iloc[:,:-4]
y_valid = valid.iloc[:,-4]


x_train = pd.DataFrame(x_train) 
x_train.to_csv('x_train.csv')
y_train = pd.DataFrame(y_train)
y_train.to_csv('y_train.csv')
x_test = pd.DataFrame(x_test)
x_test.to_csv('x_test.csv')
y_test = pd.DataFrame(y_test)
y_test.to_csv('y_test.csv')
x_valid = pd.DataFrame(x_valid)
x_valid.to_csv('x_valid.csv')
y_valid = pd.DataFrame(y_valid)
y_valid.to_csv('y_valid.csv')

