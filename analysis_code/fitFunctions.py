#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 16:35:39 2021

@author: James Bounds
"""

import scipy.interpolate
import numpy.linalg
import numpy as np

#Fitting function
def fitScales(data, basisFunctions, weights=None):
        A = [f(data[0]) for f in basisFunctions]
        if(weights is not None):
            A = np.multiply(weights,A)
            b = [np.multiply(weights,data[1])]
        else:
            b=[data[1]]
        A = np.transpose(A)
        b = np.transpose(b)
        NORM = np.max(A)
        A = A/NORM
        res = np.linalg.lstsq(A,b,rcond=None)[0]
        return (res/NORM).flatten()

#After fitting, this function makes plotting the "Total fit" much easier
def getFinalModel(basisFunctions, fitScales, minX, maxX):
    N = 3000
    interpFunctions = []
    outX = np.linspace(minX, maxX, N)
    outY = np.zeros(N)
    for i,f in enumerate(basisFunctions):
        outY += fitScales[i]*np.array(f(outX))
    return outX, outY
