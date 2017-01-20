# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:53:03 2016

@author: SROY
"""

#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/MLQuiz8bq1roc.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

import matplotlib.pyplot as plt
import numpy as np
x=[0,0,0,0,1/6,2/6,2/6,3/6,3/6,4/6,4/6,5/6,6/6]
y=[0,1/6,2/6,3/6,3/6,3/6,4/6,4/6,5/6,5/6,6/6,6/6,6/6]
#plt.xlim(x)
#plt.ylim(y)
plt.plot(x,y,'bo')
plt.plot(x,y,'b-')
plt.grid()
plt.title('Receiver operating characteristics (ROC) curve')
plt.xlabel('True Positive Rate')
plt.ylabel('True Negative Rate')
plt.xticks(np.arange(min(x), max(x)+(1/6), 1/6))
plt.yticks(np.arange(min(y), max(y)+(1/6), 1/6))
plt.show()