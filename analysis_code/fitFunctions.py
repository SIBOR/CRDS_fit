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
def fitScales(data,basisData):
    interpFunctions = []
    for i in range(len(basisData)):
        interpFunctions.append(scipy.interpolate.interp1d(basisData[i][0], basisData[i][1], fill_value = 'extrapolate'))
    A, b = [], []
    for i in range(0,len(data[0])):
        b.append(data[1][i])
        row = []
        for j in range(len(basisData)):
            row.append(interpFunctions[j](data[0][i]))
        A.append(row)
    return numpy.linalg.lstsq(A,b,rcond=None)

#Weighted fitting function
def fitScales_weighted(data,basisData,weights):
    interpFunctions = []
    for i in range(len(basisData)):
        interpFunctions.append(scipy.interpolate.interp1d(basisData[i][0], basisData[i][1], fill_value = 'extrapolate'))
    A, b = [], []
    for i in range(0,len(data[0])):
        b.append(data[1][i]*weights[i])
        row = []
        for j in range(len(basisData)):
            row.append(interpFunctions[j](data[0][i])*weights[i])
        A.append(row)
    return numpy.linalg.lstsq(A,b,rcond=None)

#After fitting, this function makes plotting the "Total fit" much easier
def getFinalModel(basisData, fitScales, minX, maxX):
    N = 3000
    interpFunctions = []
    for i in range(len(basisData)):
        interpFunctions.append(scipy.interpolate.interp1d(basisData[i][0], basisData[i][1], fill_value = 'extrapolate'))
    outX = np.linspace(minX, maxX, N)
    outY = np.zeros(N)
    for i in range(len(basisData)):
        outY += fitScales[i]*np.array(interpFunctions[i](outX))
    return outX, outY