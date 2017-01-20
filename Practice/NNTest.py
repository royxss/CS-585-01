# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:48:58 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/NNTest.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')


from sklearn.neural_network import MLPClassifier
#import numpy as np
X = [[0, 0], [0, 1], [1, 0]]
y = [[1], [0], [0]]

clf = MLPClassifier(activation='tanh',hidden_layer_sizes=(2,),solver='lbfgs')

clf.fit(X, y)
#clf.predict([[1., 2.]])
print(clf)
print(clf.coefs_)
#print(clf.intercepts_)

