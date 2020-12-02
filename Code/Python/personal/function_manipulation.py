# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:39:39 2020

@author: Jordan
"""

import numpy as np # remove this later

def repeat(f,x,n,*args,**kwargs):
    """ Applys function f n times to the input x w/ the supplied args/kwargs (not sure how to handle multiple input/output mappings but could work if wrapper fcn took x as an iterable"""

    for i in range(n):
        x = f(x,*args,**kwargs)
    return x


def apply_fcn_by_index_grouping(A,idxs,dim,f):
    """ Takes in index grouping (i.e. a lists of lists ) along dimension dim of A and applys fcn f to them
    returns list of objects (lists,arrays,etc)
    """
    
    
