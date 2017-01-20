# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 20:41:48 2016

@author: SROY
"""
#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/ParameterEstimation.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')
import matplotlib.pylab as plt
import numpy as np


def likelihood_bivar_plot(obsv_pos,obsv_total):
    """This Program plots likelihood"""
    h=obsv_pos #Number of positive observations
    t=obsv_total-h #Number of negative observations
    x=np.linspace(0, 1, num=100)
    theta=h/(h+t)
    y=np.power(x,h)*np.power(1-x,t)
    print("Total Positive Observations:",h)
    print("Total Negative Observations:",t)
    print("Maximum Liklihood value of Theta:",theta)
    plt.plot(x, y)
    #plt.ylabel('Entropy')
    plt.xlabel('Probability Theta')
    plt.show()
    
def Beta_Distribution(a, b):
    from scipy.stats import beta
    x=np.linspace(0, 1, num=100)
    print ("Mean = ",beta.mean(a, b))
    print ("Mean = ",beta.mean(b, a))
    plt.plot(beta.pdf(x, a, b))
    plt.plot(beta.pdf(x, b, a))
    plt.show()
    
def Beta_Prior(a, b):
    from scipy.stats import beta
    x=np.linspace(0, 1, num=100) #Fix this
    plt.plot(beta.pdf(x, 2+a, 3+b))
    plt.plot(beta.pdf(x, 4+a, 6+b))
    plt.plot(beta.pdf(x, 8+a, 12+b))
    plt.show() 
    
def Beta_Prior_Uniform(a):
    from scipy.stats import beta
    x=np.linspace(0, 1, num=100) #Fix this
    plt.plot(beta.pdf(x, 2+a, 3+a))
    plt.plot(beta.pdf(x, 4+a, 6+a))
    plt.plot(beta.pdf(x, 8+a, 12+a))
    plt.show()     
    
   
    