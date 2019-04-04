#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:50:30 2019

So, I created this code file to understand the purpose of certain line of python code. I was trained in Java, so I am 
learning Python at the same time I am learning neural networks.
@author: ian
"""

import numpy as np

class Network(object):
    
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        


net = Network([3,5,2])

Y = np.array([[0,0,1,1]])

        
