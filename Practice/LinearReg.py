# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 21:59:10 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/LinearReg.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

#load dataset
#select train
#select test
#select model
#fit train
#score or predict test

import matplotlib.pyplot as plt
#import seaborn
#import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.datasets import load_iris

iris=load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model=linear_model.LinearRegression()
model.fit(X_train, y_train)

print ('Accuracy = ', model.score(X_test, y_test))
print ('Predict = ', model.predict([[5.2,  2.8,  4.8,  1.8]]))
#values not coming properly

plt.scatter(X_test[:,:1], y_test)
plt.plot(X_test, model.predict(X_test))
plt.show() 