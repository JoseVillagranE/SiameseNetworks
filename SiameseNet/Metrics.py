# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 00:04:35 2020

@author: joser
"""

import numpy as np



class Metric:
    
    def __init__(self):
        pass
    
    def __call__(self, outputs, target, loss):
        raise NotImplementedError
        
    def reset(self):
        raise NotImplementedError
        
    def value(self):
        raise NotImplementedError
        
    def name(self):
        raise NotImplementedError
        

class AccumalatedAccMetric(Metric):
    
    def __init__(self):
        self.corrrect = 0
        self.total = 0
        
    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def value(self):
        return 100*float(self.correct)/self.total
    
    def name(self):
        return "Accuracy"
    

class AvgNonZeroTripletMetric(Metric):
    
    def __init__(self):
        self.values = []
        
    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()
    
    def reset(self):
        self.values = []
    
    def value(self):
        return np.mean(self.values)
    
    def name(self):
        return "Average nonzero Triplets"