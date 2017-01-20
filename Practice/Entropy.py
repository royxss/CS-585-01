#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/Entropy.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')
import matplotlib.pyplot as plt
import numpy as np
def entropy_plot():
    """This Program plots entropy"""
    x=np.linspace(0, 1, num=100)
    y=-(x*(np.log2(x)))-((np.log2(1-x))-((np.log2(1-x))*x))
    plt.plot(x, y)
    plt.ylabel('Entropy')
    plt.xlabel('Proportion')
    plt.show()

#entropy_plot()

def entropy(p):
    _entr= -(p*(np.log2(p)))-((np.log2(1-p))-((np.log2(1-p))*p))
    return _entr
#    print(_entr)
    
#entropy(0.2)    

def prob_count(c):
    _c=c/np.sum(c)
    return _c
    print(_c)

#prob_count([2,1])  

def entropy_node(p):
    n=prob_count(p)
    _n=np.array(n)*(np.log2(np.array(n)))
    _n1=np.sum(_n)*-1
    return (_n1)
#    print (_n1)

def Info_Gain_Binary(root,left,right):
    root_entropy=entropy_node(root)
    left_entropy=entropy_node(left)
    right_entropy=entropy_node(right)
    ig=root_entropy-((np.sum(left)/np.sum(root)) * left_entropy + (np.sum(right)/np.sum(root)) * right_entropy)
    return (ig)
    print(ig)
    
def Info_Gain_NonBinary(root,leaves):
    root_entropy=entropy_node(root)
    leaves_entropy=[]
    leaves_prob=[]
    for ls in leaves:
        leaves_entropy.append(entropy_node(ls))
        leaves_prob.append(np.sum(ls))
    exp_leaves_entropy=leaves_prob/np.sum(root) * leaves_entropy
    print(exp_leaves_entropy)
    ig=root_entropy-(np.sum(exp_leaves_entropy))
    return(ig)
    print(ig)    
          