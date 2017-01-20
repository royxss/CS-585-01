# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 11:29:40 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz7bLR.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

#load dataset
#select train
#select test
#select model
#fit train
#score or predict test

#Import packages
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

#Load data
bc=load_breast_cancer()
X, y = bc.data, bc.target

#Select data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=1)

#Select and fit Model
model=LogisticRegression()
model.fit(X_train, y_train)

#print ('Model Score = ', model.score(X_test, y_test))
#print ('Model Predict = ', model.predict(X_test))
#print ('Decision Function = ', model.decision_function(X_test))
#print ('Predict Proba = ', model.predict_proba(X_test))

y_true, y_pred, y_score  = y_test, model.predict(X_test), model.decision_function(X_test) 

print ('Accuracy = %0.3f' %metrics.accuracy_score(y_true, y_pred))
print ('Precision = %0.3f' %metrics.precision_score(y_true, y_pred))
print ('Recall = %0.3f' %metrics.recall_score(y_true, y_pred))
print ('AUC = %0.3f' %metrics.roc_auc_score(y_true, y_score))
