# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 10:41:10 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz8bq2cv.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

#Import packages
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
#from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import numpy as np

#Load data
bc=load_breast_cancer()
X, y = bc.data, bc.target

#Select data
#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, random_state=1)

#Select and fit Model
model=LogisticRegression()
#model.fit(X_train, y_train)

accuracy = cross_val_score(model, X, y, cv=6, scoring='accuracy') 
#print('accuracy Array : ',accuracy)
print('accuracy Mean = %0.3f' %np.mean(accuracy))

precision = cross_val_score(model, X, y, cv=6, scoring='precision') 
#print('precision Array : ',precision)
print('precision Mean = %0.3f' %np.mean(precision))

recall = cross_val_score(model, X, y, cv=6, scoring='recall') 
#print('recall Array : ',recall)
print('recall Mean = %0.3f' %np.mean(recall))

roc_auc = cross_val_score(model, X, y, cv=6, scoring='roc_auc') 
#print('roc_auc Array : ',roc_auc)
print('roc_auc Mean = %0.3f' %np.mean(roc_auc))


#print ('Model Score = ', model.score(X_test, y_test))
#print ('Model Predict = ', model.predict(X_test))
#print ('Decision Function = ', model.decision_function(X_test))
#print ('Predict Proba = ', model.predict_proba(X_test))

#y_true, y_pred, y_score  = y_test, model.predict(X_test), model.decision_function(X_test) 

#print ('Accuracy = %0.3f' %metrics.accuracy_score(y, mean_predicted))
#print ('Precision = %0.3f' %metrics.precision_score(y, predicted))
#print ('Recall = %0.3f' %metrics.recall_score(y, predicted))
#print ('AUC = %0.3f' %metrics.roc_auc_score(y, predicted))