# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 20:34:12 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz6bGradOpt.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

import numpy as np
from scipy.optimize import minimize

def f(w):
    return 6*w-(11*(np.log(1+np.exp(w))))

def fp(w):
    return 6-(11*(np.exp(w)/(1+np.exp(w))))

x0=1

func=lambda w: -f(w)
deriv=lambda w: -fp(w)

res = minimize(func, x0, method='BFGS', jac=deriv, options={'disp': True}, tol=1e-6)

res.x
print (res.message)