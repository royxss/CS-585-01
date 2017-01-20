# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 19:51:21 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz6bGNB.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

#load dataset
#select train
#select test
#select model
#fit train
#score or predict test

#Import packages
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#Load data
bc=load_breast_cancer()
X, y = bc.data, bc.target

#Select data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=0)

#Select and fit Model
model=GaussianNB()
model.fit(X_train, y_train)

print ('Accuracy = ', model.score(X_test, y_test))