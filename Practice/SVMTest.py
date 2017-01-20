# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:49:40 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/SVMTest.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

from sklearn.svm import SVC
import numpy as np

X = np.matrix('1 1; 2 3; 2.5 3; 3 1')
y = np.matrix('1;1;-1;-1')
#a = np.matrix('0;8;8;0')

clf=SVC(kernel='linear')
print('Classifier Details: ',clf.fit(X,y))
print('get weights: ',clf.coef_)  #Only for linear
print('get support vectors: ',clf.support_vectors_)
print('get indices of support vectors: ',clf.support_)
print('get number of support vectors for each class: ',clf.n_support_)

print('Prediction: ',clf.predict([[0., 0.]]))