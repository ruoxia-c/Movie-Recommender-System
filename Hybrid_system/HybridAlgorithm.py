#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:28:11 2021

@author: ruoxia
"""

from surprise import AlgoBase

class HybridAlgorithm(AlgoBase):

    def __init__(self, algorithms, weights, sim_options={}):
        AlgoBase.__init__(self)
        self.algorithms = algorithms
        self.weights = weights #weights for each algorithm

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        for algorithm in self.algorithms:
            algorithm.fit(trainset) #train each algorithm
                
        return self

    def estimate(self, u, i):
        
        sumScores = 0
        sumWeights = 0
        
        for idx in range(len(self.algorithms)):
            sumScores += self.algorithms[idx].estimate(u, i) * self.weights[idx]
            sumWeights += self.weights[idx]
            
        return sumScores / sumWeights