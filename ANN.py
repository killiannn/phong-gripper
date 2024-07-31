import pandas as pd


import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from tabgan.sampler import  GANGenerator

# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

X_train = pd.read_csv('x_train.csv', header = 0, index_col=0)
Y_train = pd.read_csv('y_train.csv', header=0, index_col=0)
x_test = pd.read_csv('x_test.csv', header=0, index_col=0)
y_test = pd.read_csv('y_test.csv', header=0, index_col=0)
x_valid = pd.read_csv('x_valid.csv', header=0, index_col=0)
y_valid = pd.read_csv('y_valid.csv', header=0, index_col=0)
x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

scaler = StandardScaler()  
scaler.fit(X_train)  
x_train = scaler.transform(X_train)
x_test = scaler.transform(x_test)

mlp = MLPRegressor(solver='adam', alpha=1e-5, max_iter= 10000,
                hidden_layer_sizes=(10), random_state=1,
                activation= 'logistic', early_stopping=True, n_iter_no_change=10)
mlp.fit(x_train,Y_train)

def test():
    predict_test = mlp.predict(x_test)
    # y_test = y_test.to_numpy()
    # print(predict_test)
    # print(y_test)

    mse_test = 0
    mse_test =  mean_squared_error(y_test, predict_test)
    print('mse: ', mse_test)

    r2_test = 0
    r2_test = r2_score(y_test, predict_test)
    print('r2: ', r2_test)

    mape_test = 0
    mape_test =  mean_absolute_percentage_error(y_test, predict_test)
    print('mape: ',mape_test)

def valid():
    predict_valid = mlp.predict(x_valid)
    # y_test = y_test.to_numpy()
    # print(predict_test)
    # print(y_test)

    mse_valid = 0
    mse_valid =  mean_squared_error(y_valid, predict_valid)
    print('mse: ', mse_valid)

    r2_valid = 0
    r2_valid = r2_score(y_valid, predict_valid)
    print('r2: ', r2_valid)

    mape_valid = 0
    mape_valid =  mean_absolute_percentage_error(y_valid, predict_valid)
    print('mape: ',mape_valid)

    

test()
valid() 