# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:48:58 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz10bNN.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

df=load_breast_cancer()
X, y = df.data, df.target

clf=MLPClassifier(random_state=2)
accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy') 
print('accuracy Array : ',accuracy)
print('accuracy Mean = %0.3f' %np.mean(accuracy))
