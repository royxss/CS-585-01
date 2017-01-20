# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 17:43:30 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/matrices.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

import numpy as np

a=np.matrix('1 2;3 4')
b=np.matrix('1 2;3 4')
print('Matrix A = ',a)
print('Matrix B = ',b)

print('Matrix Sum A+B = ',a+b)
print('Transpose of A = ', np.transpose(a))

#Calculate inverse numpy.linalg.inv(x)