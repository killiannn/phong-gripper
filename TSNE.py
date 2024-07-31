import plotly.express as px
import pandas as pd
# from sklearn.datasets import make_classification
import numpy as np

df = pd.read_csv('C:\Pro\ANN\phong_gripper.csv', header= 0, index_col=0)  # type: ignore
df.astype('float')

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
            er.append(0.01)


    for b in range(len(x_test)):
        if np.array_equal(x[i] , x_test[b]) == True:
            type.append(2)
            er.append(0.01)

    for c in range(len(x_valid)):
        if np.array_equal(x[i] , x_valid[c]) == True:
            type.append(3)
            for d in range(len(error)):
                if error['0'][d] == y_valid[c]:
                    er.append(error['error'][d]*2)
                

print(type)
print(er)

# print(x)
# print(y)
# df['type'] = type
# print(df.head())

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0 , perplexity=15)
X_tsne = tsne.fit_transform(x)
tsne.kl_divergence_
# print(X_tsne)

fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1] , symbol = type, color = type, size = er )
fig.update_layout(
    title="t-SNE visualization of Custom Classification dataset",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)    

fig.show()