#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 23:28:48 2021

@author: ruoxia
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator

import random
import numpy as np

def LoadMovieLensData():
    ml = MovieLens()
    print("Loading movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

(ml, evaluationData, rankings) = LoadMovieLensData()

evaluator = Evaluator(evaluationData, rankings)

#Simple RBM
SimpleRBM = RBMAlgorithm(epochs=1) #40
#Content
ContentKNN = ContentKNNAlgorithm()

#Combine them
Hybrid = HybridAlgorithm([SimpleRBM, ContentKNN], [0.5, 0.5])

evaluator.AddAlgorithm(SimpleRBM, "RBM")
evaluator.AddAlgorithm(ContentKNN, "ContentKNN")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(False)

evaluator.SampleTopNRecs(ml)
