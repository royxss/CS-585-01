# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 08:50:00 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/plotgraph.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

import numpy as np
from matplotlib import pyplot as plt

def f(x):
    return np.power(x,3)+ 2*np.power(x,2) + np.sin(1.5*x)

x=np.linspace(0,5,100)
y=f(x)
plt.plot(x,y,color="blue", label="What???") 
plt.legend(loc='lower right')   