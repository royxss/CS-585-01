# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 18:54:56 2016

@author: SROY
"""
#runfile('C:/Users/SROY/Documents/CodeBase/PythonWorkspace/regex_op.py', wdir='C:/Users/SROY/Documents/CodeBase/PythonWorkspace')

def split_str(name,delim):
    #splits string on delims and stores it into array
    import re
    lst=re.split(delim,name)
    print("split_str : ",lst)
    return lst
   
def rmv_dups():
    #remove duplicates from string delims
    #use set to remove duplicates
    l=[set(split_str("a;b;c;d;a",";"))]
    print("Original String : a;b;c;d;a")
    print("rmv_dups : ",l)
    return l
    
def srt_lst():
    #Sorts array
    l1=rmv_dups()
    print("srt_lst : ",sorted(l1))
    

