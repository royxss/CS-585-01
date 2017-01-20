# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:29:21 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz9bLasso.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import numpy as np

df=load_boston()
X, y = df.data, df.target

model=Lasso()
nmse = cross_val_score(model, X, y , cv=7, scoring='neg_mean_squared_error')
nrmse=np.mean(nmse)
print("Negative Mean Squared Error : ", nmse)
print("Average Mean Squared Error : %0.1f" %nrmse)