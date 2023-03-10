"""
Implements a simple binary decision tree.
The tree makes a binary classification.
"""

import numpy as np

class DecisionTree:
    def __init__(self, max_level):
        self.max_level = max_level


class TreeNode:
    def __init__(self, dimension, max_lvl):
        self.dim = dimension
        self.max_lvl = max_lvl
    
    def fit(self,X,Y):
        """
        X: (n,dim) n datapoints in dim dimensional space
        Y: (n) binary array with values 0,1
        """
        avg = np.mean(Y)
        n = X.shape[0]

        # check if we have reached max depht
        if(self.max_lvl <= 0):
            self.leaf = True
            # the classification will simply be the majority of datapoints
            self.value = np.round(avg)
            return
        
        #check if the population is homogenous
        ones = np.sum(Y==1.0)
        if ones==n or ones==0:
            self.leaf = True
            # the classification will simply be the majority of datapoints
            self.value = np.round(avg)
        
        # now if the population is not homogenous and we can still build leaves
        # add a new decision rule to the tree
    
    def find_best_split(x,y):
        order = np.argsort(x)
        #TODO