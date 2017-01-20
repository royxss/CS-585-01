# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 12:17:30 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz9bWeights.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

import numpy as np


data=np.matrix('2 3 2; 0 2 4; 4 0 3; 3 4 0; 1 1 1')
y=np.matrix('11; 7; 11.25; 13; 5.25')
X=np.hstack((np.ones([data.shape[0],1]),data))

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


