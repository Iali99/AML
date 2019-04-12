import numpy as np
import pandas as pd
import fileinput
import csv
from scipy import stats
from matplotlib import pyplot as plt

def mylinridgereg(X,Y,l):
    Xt = np.transpose(X)
    I = np.identity(X.shape[1])
    W = np.dot(np.matmul(np.linalg.inv(np.matmul(Xt,X) + l*I),Xt), Y)
    # print(W)
    return W
def mylinridgeregeval(X,W):
    y = np.dot(X,W)
    return y
def meansquarederr(T, Tdash):
    mse = (np.square(T - Tdash)).mean()
    return mse

def mainFunction():

    data = []
    with open('linregdata', 'rU') as f:  #opens PW file
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=','))
        for i in range(len(data)):
            item = data[i]
            if item[0] == 'M':
                item = [0,0,1]+ item[1:]
            if item[0] == 'F':
                item = [1,0,0]+ item[1:]
            if item[0] == 'I':
                item = [0,1,0]+ item[1:]
            data[i] = item

    dataset = np.asarray(data).astype(float)
    # print(dataset)
    # print(dataset[:,-1])
    # print(dataset.shape[1])



    # print(dataset)
    X = dataset[:,:-1]
    # print(X.shape[1])
    # for i in range(0,X.shape[1]):
    #     mean = np.average(X[:,i])
    #     dev = np.std(X[:,i])
    #     for item in X:
    #         item[i] - (item[i] - mean)/dev
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # print(X[0])
    X = np.insert(X,0,1,axis=1)
    Xtrain = X[:int(0.8*len(X)),:]
    Xtest = X[int(0.8*len(X)):,:]
    # Xtrain = np.array([x for i,x in enumerate(X) if i%5 != 0])
    # Xtest = np.array([x for i,x in enumerate(X) if i%5 == 0])
    # print(X)
    y = dataset[:,-1]
    ytrain = y[:int(0.8*len(y))]
    ytest = y[int(0.8*len(y)):]
    # ytrain = np.array([x for i,x in enumerate(y) if i%5 != 0])
    # ytest = np.array([x for i,x in enumerate(y) if i%5 == 0])
    # print(y)
    l = np.array([0.001,0.01,0.1]+ list(range(150)))
    print(l)
    error = []
    for lamda in l:
        W = mylinridgereg(Xtrain,ytrain,lamda)
        # print(W)
        y1 = mylinridgeregeval(Xtest,W)
        # print(y1)
        # print(y[1])
        error.append(meansquarederr(y1, ytest))
    print(error)
    plt.plot(l,error)
    plt.show()
    # print((np.mean(dataset,axis=0)).astype(int))


mainFunction()
# print(meansquarederr(np.array([1,2,3]), np.array([1,2,3])))
