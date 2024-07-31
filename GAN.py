import pandas as pd
import numpy as np 

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from tabgan.sampler import  GANGenerator

# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

# df = pd.read_csv('C:\Pro\ANN\phong_gripper.csv', header= 0, index_col=0)  # type: ignore
# df.astype('float')
# rng = RandomState()

def ANN(X,Y):
    X_train = pd.read_csv('x_train.csv', header = 0, index_col=0)
    Y_train = pd.read_csv('y_train.csv', header=0, index_col=0)
    x_test = pd.read_csv('x_test.csv', header=0, index_col=0)
    y_test = pd.read_csv('y_test.csv', header=0, index_col=0)
    x_valid = pd.read_csv('x_valid.csv', header=0, index_col=0)
    y_valid = pd.read_csv('y_valid.csv', header=0, index_col=0)
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    # model_data = df.sample(frac=0.8, random_state=rng)
    # valid = df.loc[~df.index.isin(model_data.index)]

    # x_train,x_test,y_train,y_test = train_test_split(model_data.iloc[:,:-4], 
    # model_data.iloc[:,-4],test_size = 0.3)

    # x_valid = valid.iloc[:,:-4]
    # y_valid = valid.iloc[:,-4]

    # x_test = x_test.to_numpy()
    # y_test = y_test.to_numpy()
    # X_train = pd.DataFrame(x_train)
    # Y_train = pd.DataFrame(y_train)
    test = pd.DataFrame(np.random.randint(0, 10, size=(int(X), 4)), columns=X_train.columns)

    # GAN
    new_train2, new_target2 = GANGenerator(gen_x_times= int(Y), cat_cols=None,bot_filter_quantile= 0, top_filter_quantile= 1,
                is_post_process=False,
            adversarial_model_params={
                "metrics": "rmse", "max_depth": 2, "max_bin": 100, 
                "learning_rate": 0.02, "random_state": 42, "n_estimators": 500,
            }, pregeneration_frac=1,only_generated_data= False,gen_params={"batch_size": 500, "epochs": 500, "patience": 25 }).generate_data_pipe(X_train, Y_train, test, only_adversarial=False, deep_copy=True,use_adversarial=True )
    print(new_train2, new_target2)

    # MODEL 

    scaler = StandardScaler()  
    scaler.fit(new_train2)  
    x_train = scaler.transform(new_train2)
    x_test = scaler.transform(x_test)

    mlp = MLPRegressor(solver='adam', alpha=1e-5, max_iter= 10000,
                    hidden_layer_sizes=(10), random_state=1,
                    activation= 'logistic', early_stopping=True, n_iter_no_change=10)
    mlp.fit(x_train,new_target2)

    # PREDICT
    predict_test = mlp.predict(x_test)
    
    # y_test = y_test.to_numpy()
    # print(predict_test)
    # print(y_test)

    # error =[]
    # for i in range(len(predict_train)):
    #     a = abs(new_target2[i] - predict_train[i])**2
    #     error.append(a)

    # for z in range(len(predict_test)):
    #     b = abs(y_test[z] - predict_test[z])**2
    #     error.append(b)

    # for u in range(len(predict_valid)):
    #     c = abs(y_valid[u] - predict_valid[u])**2
    #     error.append(c)
    
    
    # print(error)

    # error = pd.DataFrame(error)
    # error.to_csv('error.csv')
    # new_train = pd.DataFrame(new_train2)
    # new_train.to_csv('new_train.csv')
    # new_target = pd.DataFrame(new_target2)
    # new_target.to_csv('new_target.csv')

    mse_test = 0
    mse_test =  mean_squared_error(y_test, predict_test)
    print('mse: ', mse_test)
    
    # for i in range(len(predict_test)):
    #     mse_test = mse_test + mean_squared_error(predict_test[i], y_test[i])
    # mse_test = mse_test / len(predict_test)
    # print('mse: ', mse_test)

    # # for i in range(len(predict_test)):
    # #     r2_test = r2_test + r2_score(predict_test[i], y_test[i])
    # # r2_test = r2_test / len(predict_test)
    # print('r2: ', r2_test)
    
    # # for i in range(len(predict_test)):
    # #     mape_test = mape_test + mean_absolute_percentage_error(predict_test[i], y_test[i])
    # # mape_test = mape_test / len(predict_test)
    r2_test = 0
    r2_test = r2_score(y_test, predict_test)
    print('r2: ', r2_test)

    mape_test = 0
    mape_test =  mean_absolute_percentage_error(y_test, predict_test)
    print('mape: ',mape_test)
    # return r2_test

    return mse_test
    # VALID => FUNCTION SELECTION

    # predict_valid = mlp.predict(x_valid)
    # # print(predict_valid)
    # # print(y_valid)

    # mse_valid = 0
    # mse_valid =  mean_squared_error(y_valid, predict_valid)
    # print('mse: ', mse_valid)
    # return mse_valid

    # r2_valid = 0
    # r2_valid = r2_score(y_valid, predict_valid)
    # print('r2: ', r2_valid)

    # for i in range(len(predict_valid)):
    #     r2_valid = r2_valid + r2_score(predict_valid[i], y_valid[i])
    # r2_valid = r2_valid / len(predict_valid)

    # mape_valid = 0
    # mape_valid =  mean_absolute_percentage_error(predict_valid, y_valid)
    # # for i in range(len(predict_valid)):
    # #     mape_valid = mape_valid + mean_absolute_percentage_error(predict_valid[i], y_valid[i])
    # # mape_valid = mape_valid / len(predict_valid)
    # print('mape: ',mape_valid)

def ANN_valid(X,Y):
    X_train = pd.read_csv('x_train.csv', header = 0, index_col=0)
    Y_train = pd.read_csv('y_train.csv', header=0, index_col=0)
    x_test = pd.read_csv('x_test.csv', header=0, index_col=0)
    y_test = pd.read_csv('y_test.csv', header=0, index_col=0)
    x_valid = pd.read_csv('x_valid.csv', header=0, index_col=0)
    y_valid = pd.read_csv('y_valid.csv', header=0, index_col=0)

    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()
    # model_data = df.sample(frac=0.8, random_state=rng)
    # valid = df.loc[~df.index.isin(model_data.index)]

    # x_train,x_test,y_train,y_test = train_test_split(model_data.iloc[:,:-4], 
    # model_data.iloc[:,-4],test_size = 0.3)

    # x_valid = valid.iloc[:,:-4]
    # y_valid = valid.iloc[:,-4]

    # x_test = x_test.to_numpy()
    # y_test = y_test.to_numpy()
    # X_train = pd.DataFrame(x_train)
    # Y_train = pd.DataFrame(y_train)
    test = pd.DataFrame(np.random.randint(0, 100, size=(int(X), 4)), columns=X_train.columns)

    # GAN
    new_train2, new_target2 = GANGenerator(gen_x_times= int(Y), cat_cols=None,bot_filter_quantile= 0, top_filter_quantile= 1,
                is_post_process=False,
            adversarial_model_params={
                "metrics": "rmse", "max_depth": 2, "max_bin": 100, 
                "learning_rate": 0.02, "random_state": 42, "n_estimators": 500,
            }, pregeneration_frac=1,only_generated_data= False,gen_params={"batch_size": 500, "epochs": 500, "patience": 25 }).generate_data_pipe(X_train, Y_train, test, only_adversarial=False, deep_copy=True,use_adversarial=True )
    print(new_train2, new_target2)

    # MODEL 

    scaler = StandardScaler()  
    scaler.fit(new_train2)  
    x_train = scaler.transform(new_train2)
    x_test = scaler.transform(x_valid)

    mlp = MLPRegressor(solver='adam', alpha=1e-5, max_iter= 10000,
                    hidden_layer_sizes=(10), random_state=1,
                    activation= 'logistic', early_stopping=True, n_iter_no_change=10)
    mlp.fit(x_train,new_target2)

    # PREDICT

    # VALID => FUNCTION SELECTION

    predict_valid = mlp.predict(x_valid)
    y_valid = y_valid.to_numpy()
    print(predict_valid)
    print(y_valid)

    error =[]
    for i in range(len(predict_valid)):
        a = abs(y_valid[i] - predict_valid[i])**2
        error.append(a)
    print(error)

    y_val = pd.DataFrame(y_valid)
    y_val['error'] = error
    error = pd.DataFrame(y_val) 
    
    error.to_csv('error.csv')

    mse_valid = 0
    mse_valid =  mean_squared_error(y_valid, predict_valid)
    print('mse: ', mse_valid)
    
    r2_valid = 0
    r2_valid = r2_score(y_valid, predict_valid)
    print('r2: ', r2_valid)

    # for i in range(len(predict_valid)):
    #     r2_valid = r2_valid + r2_score(predict_valid[i], y_valid[i])
    # r2_valid = r2_valid / len(predict_valid)

    mape_valid = 0
    mape_valid =  mean_absolute_percentage_error(predict_valid, y_valid)
    # for i in range(len(predict_valid)):
    #     mape_valid = mape_valid + mean_absolute_percentage_error(predict_valid[i], y_valid[i])
    # mape_valid = mape_valid / len(predict_valid)
    print('mape: ',mape_valid)

    return mse_valid



