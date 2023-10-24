#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:24:55 2021

@author: jaymz
"""

import numpy as np
from scipy.stats import kde
import scipy.signal

#input data must be evenly sampled!
def localMaxMin(x,y,windowSize = 13, polyOrder = 2):
    yDeriv = scipy.signal.savgol_filter(y,windowSize, polyOrder, deriv=1)
    maxs, mins =[], []
    for i in range(len(yDeriv)-1):
        if(yDeriv[i]*yDeriv[i+1] <= 0): #zero cross detect
            xzero = yDeriv[i]*(x[i] - x[i+1])/(yDeriv[i]-yDeriv[i+1]) + x[i]
            if(yDeriv[i+1] > yDeriv[i]): #positive slope -> local min
                mins.append(xzero)
            else:
                maxs.append(xzero)
    return mins,maxs

def splitData(x,splitPoints):
    if(len(splitPoints) == 0):
        return [x]
    splits = []
    s0 = splitPoints[0]
    mask=x<s0
    splits.append(x[mask])
    for i in range(0,len(splitPoints)-1):
        s1,s2 = splitPoints[i], splitPoints[i+1]
        mask =  x >= s1
        mask *= x <  s2
        splits.append(x[mask])
    s0 = splitPoints[-1]
    mask = x>=s0
    splits.append(x[mask])
    return splits

def findClustersAndSplit(x):   
    density = kde.gaussian_kde(x)
    xgrid = np.linspace(x.min(), x.max(), 1000)
    mns, mxs = localMaxMin(xgrid,density(xgrid))
    return splitData(x,mns)

def clusterAvg(x,cutoffSize = 1):
    clusts = findClustersAndSplit(x)
    lens = []
    outAvgs = []
    for c in clusts:
        if(len(c) < cutoffSize):
            continue
        outAvgs.append(np.mean(c))
        lens.append(len(c))
    #print(outAvgs)
    #print(lens)
    return outAvgs, lens