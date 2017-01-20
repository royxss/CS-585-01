# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 11:00:38 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz9bRidge.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')


from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge

df=load_boston()
X_train, y_train = df.data, df.target
X_test=[[4.74410435e-01, 1.13498024e+01, 9.50316206e+00, 7.90513834e-02, 5.20696838e-01, 6.33158498e+00, 6.57134387e+01, 4.15064269e+00, 4.56521739e+00, 3.20920949e+02, 1.79158103e+01, 3.76613123e+02, 1.13721344e+01]]

model=Ridge()
model.fit(X_train, y_train)
print(model.get_params)

y_predict = model.predict(X_test)

print("Predict : %0.1f" %y_predict)


