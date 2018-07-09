from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from zlib import crc32


def array_str_to_float(X):
    for x1 in range(X.shape[0]):
        for x2 in range(X.shape[1]):
            if isinstance(X[x1][x2], str):
                X[x1][x2] = float(crc32(X[x1][x2]) & 0xffffffff) / 2**32
    return X


def svr_predict(X):
    y_predict = []
    for test_index in range(len(X)):
        y_predict.append(clf.predict([X[test_index]])[0])
    return y_predict

# Load data
train_data = pd.read_csv('DataPre/train_data.csv')
train_data_fillna = train_data.fillna(train_data.mean())
nunique = train_data_fillna.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique <= 1].index
train_drop_duplicate = train_data_fillna.drop(cols_to_drop, axis=1)

X = train_drop_duplicate.values[:, 2:(train_drop_duplicate.shape[1]-1)]
y = train_drop_duplicate.values[:, train_drop_duplicate.shape[1]-1]

X_trans_str = array_str_to_float(X)
X_scaled = preprocessing.scale(X_trans_str)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)
clf = svm.SVR(gamma=0.00001)
clf.fit(X_train, y_train)
y_test_predict = svr_predict(X_test)
y_train_predict = svr_predict(X_train)

MSE_test = mean_squared_error(y_test_predict, y_test)
MSE_train = mean_squared_error(y_train_predict, y_train)
print y_test
print MSE_test, MSE_train

# Predict
test_A = pd.read_csv('DataPre/test_A.csv')
test_A_data_fillna = test_A.fillna(train_data.mean())
test_A_drop_duplicate = test_A_data_fillna.drop(cols_to_drop, axis=1)
X_A = test_A_drop_duplicate.values[:, 2:(test_A_drop_duplicate.shape[1])]
X_A_trans_str = array_str_to_float(X_A)
X_A_scaled = preprocessing.scale(X_A_trans_str)
y_A_predict = svr_predict(X_A_scaled)

format_A = my_data = pd.read_csv('DataPre/testA_format.csv',header=None)
output = pd.concat([format_A, pd.DataFrame(y_A_predict)], axis=1)
output.to_csv('output/test_A_1218.csv',header=False,index=False)
