# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 08:55:13 2016

@author: SROY
"""


#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/linReg.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

import numpy as np

def f(x):
    return 1+2*x
'''
def construct_A(X):
    A=[]
    for i in range(X.shape[1]):
        row = []
        for j in range(X.shape[1]):
            s=0
            for d in X:
                s += d[i]*d[j]
            row.append(s)
        A.append(row)
    print (np.matrix(A))
    #return np.matrix(A)   

data=np.matrix('2; 3')
y=np.matrix('5; 7')
X=np.hstack((np.ones([data.shape[0],1]),data))
print(X)
construct_A(X)'''
    

#data=np.matrix('2;4')
y=np.matrix('5;7;10;14')
#X=np.hstack((np.ones([data.shape[0],1]),data))
X=np.matrix('2;3;6;4')

#Method 1: Using linear model
from sklearn.linear_model import LinearRegression
model= LinearRegression(fit_intercept=False)
model.fit(X,y)

print('X : \n',X)
print('y : \n',y)
print ('Regression Coefficients: \n', model.coef_)


#Method 2: Usinglinear algebra
print ('\nList Squares: \n')
print(np.linalg.lstsq(X,y))

#Method 3: using manual calc

wt=(np.linalg.inv(X))*y
print("\nWeights : \n", wt)