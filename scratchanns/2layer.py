#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:51:34 2019

@author: ian
"""

import numpy as np

#compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output


#convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

#input dataset
X = np.array([[0,1],[0,1],[1,0],[1,0]])

#output dataset
Y = np.array([[0,0,1,1]])
y = np.array([[0,0,1,1]]).T

#seed random numbers to make calculation
# deterministic (just for good practice)
np.random.seed(1)

#initialize weights randomly with mean 0
synapse_0 = 2*np.random.random((2,1)) - 1

for iter in range(10000):
    
    #forward propagation
    layer_0 = X
    layertest = np.dot(layer_0,synapse_0)
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))
    
    #how much did we miss?
    layer_1_error = layer_1 - y
    
    #These two lines are used for test purpose so I can trace the calculations
    sigmoidoutputtoderivative = sigmoid_output_to_derivative(layer_1)
    layer0transpose = layer_0.T
    
    #multiply how much we missed by the slope of the sigmoid at the values in l1
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    synapse_0_derivative = np.dot(layer_0.T,layer_1_delta)
    
    #update weights
    synapse_0 -= synapse_0_derivative
    
print ("Output After Training:")
print (layer_1) 