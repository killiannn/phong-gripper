import plotly.express as px
import pandas as pd
# from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:\Pro\ANN\phong_gripper.csv', header= 0, index_col=0)  # type: ignore
df.astype('float')
df.columns = df.columns.str.strip()

x = df.iloc[:,:-4]
y = df.iloc[:,-4]
x= x.to_numpy()
y = y.to_numpy()    
y = y.flatten()

x_train = pd.read_csv('x_train.csv', header = 0, index_col=0)
y_train = pd.read_csv('y_train.csv', header=0, index_col=0)
x_test = pd.read_csv('x_test.csv', header=0, index_col=0)
y_test = pd.read_csv('y_test.csv', header=0, index_col=0)
x_valid = pd.read_csv('x_valid.csv', header=0, index_col=0)
y_valid = pd.read_csv('y_valid.csv', header=0, index_col=0)

error = pd.read_csv('error.csv', header=0, index_col=0)
def extract_float(error_str):
    # Remove the brackets and extract the number
    number_str = error_str.strip('[]')
    # Convert the string to a float
    return float(number_str)

# Apply the function to the 'error' column
error['error'] = error['error'].apply(extract_float)
print(error)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
x_valid = x_valid.to_numpy()
y_valid = y_valid.to_numpy()
# error = error.to_numpy()

type = []
er =[]
# for i in range(len(x)):
for i in range(len(x)):
    for a in range(len(x_train)):
        if np.array_equal(x[i] , x_train[a]) == True:
            type.append(1)
            er.append(0.0025)

    for b in range(len(x_test)):
        if np.array_equal(x[i] , x_test[b]) == True:
            type.append(2)
            er.append(0.00125)

    for c in range(len(x_valid)):
        if np.array_equal(x[i] , x_valid[c]) == True:
            type.append(3)
            for d in range(len(error)):
                if error['0'][d] == y_valid[c]:
                    er.append(error['error'][d])
                

print(type)
print(er)

# print(x)
# print(y)
# df['type'] = type
# print(df.head())


fig = px.scatter(x=df['x1 (mm)'], y=df['x3 (mm)'] , symbol = type, color = type, size = er, opacity = 0.7, size_max = 40 )
fig.update_layout(
    title="Correlation-based dimensionality reduction",
    xaxis_title="x1 (mm)",
    yaxis_title="x3 (mm)",
)    
fig.update_traces(marker = dict(line = dict(width = 2,
                                            color = 'DarkSlateGrey')),
                selector=dict(mode='markers')
)
fig.show()

# for i in range(len(er)):
#     er[i] = er[i]*10000

# markers = {'1': 'o', '2': 's', '3': '^'}
# fig , ax = plt.subplots()
# scatter = ax.scatter(df['x1 (mm)'], df['x3 (mm)'], c=type, edgecolor='k', alpha=0.7 , s = er)
# plt.title(f'Scatter Plot of x1 vs x3')
# plt.xlabel('x1')
# plt.ylabel('x3')
# plt.grid(True)
# legend1 = ax.legend(*scatter.legend_elements(),
#                     loc="lower left", title="Classes")
# ax.add_artist(legend1)
# plt.show()