#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:03:28 2021

@author: jaymz
"""

from .molecules import molecules
from . import hitranPlotModule as hpm
from .fitFunctions import fitScales, fitScales_weighted, getFinalModel
from .external_xs.acetone import acetone_nir_xs

import numpy as np
import scipy.stats.mstats


c_cm = 299792458*100  #Speed of light in cm/sec

class CRDS_Fit:
    def __init__(self,tempData=None, currData=None, wlData = None, rdtData=None, wlCoeffs = None):
        self.molecules = molecules
        self.hapi = hpm.hapi
        
        #Wavelength range for use in HITRAN simulation
        self.wl_start=1640.0
        self.wl_end = 1660.0

        #For fitting things not in the HITRAN database
        self.external_cross_sections = {'acetone' : acetone_nir_xs}
        
        #Vacuum data and fit
        self.vac_fit = None
        self.vac_data = None

        #Main data storage
        if(tempData is not None):
            self.data_temp = np.array(tempData)
        if(currData is not None):
            self.data_curr = np.array(currData)
        if(rdtData is not None):
            self.data_rdt  = np.array(rdtData)
        
        #Ambient conditions
        self.temperature = None
        self.pressureAtm = None
        
        #HITRAN parameters and curves
        self.mols = None        #Molecule names
        self.isos = None        #Iosotopologue id
        self.basisData = None   #list of [x,y] sets where x and y are arrays of points
        self.aux_indices = None #Used to fit non-HITRAN things
        
        #Fit results
        self.scaleFit = None
        
        #Wavelength information
        self.data_wl = None
        self.wl_coeffs = wlCoeffs
        if(wlCoeffs is not None):
            self.data_wl = wlCoeffs[0] + \
                           wlCoeffs[1]*self.data_curr + \
                           wlCoeffs[2]*self.data_temp +  \
                           wlCoeffs[3]*self.data_curr**2 + \
                           wlCoeffs[4]*self.data_temp**2 + \
                           wlCoeffs[5]*self.data_temp*self.data_curr
        if(wlData is not None):
            self.data_wl = wlData
    
    def setVacuum(self,vacX,vacY):
        self.vac_data = [vacX,vacY]
        self.vac_fit = scipy.stats.mstats.theilslopes(vacY,vacX)
    def vacuumRingdown(self,x):
        if(self.vac_fit is None):
            print("No vacuum data fit yet!!!")
            return
        return x*self.vac_fit[0] + self.vac_fit[1]
    
    def rdtFromAlpha(self, alpha, wavelength):
        return 1.0/(alpha*c_cm + 1.0/self.vacuumRingdown(wavelength))
    
    def computeHitranBasis(self,mols,isos,temperature,pressure_atm, fit_offset = True, external_fit = []):
        self.temperature=temperature
        self.pressureAtm = pressure_atm
        
        hpm.wl_start = self.wl_start
        hpm.wl_end   = self.wl_end

        crossSections = hpm.computeCrossSections(mols,isos,temperature,pressure_atm)
        
        basisData = []
        self.aux_indices = {}
        
        nDensity = hpm.nDensity(temperature,pressure_atm)
        
        for i in range(len(crossSections)):
            basisData.append([crossSections[i][0],
                              crossSections[i][1]*nDensity])
        if(fit_offset):
            offsetBasis = np.array([[self.wl_start,self.wl_end],[1.0,1.0]])
            basisData.append(offsetBasis)
            self.aux_indices['offset'] = len(basisData) - 1
        for ef in external_fit:
            if( ef in self.external_cross_sections ):
                interpFunc = self.external_cross_sections[ef]
                #x_extern = np.linspace(self.wl_start, self.wl_end, len(crossSections[0][0]))
                x_extern = np.linspace(self.wl_start, self.wl_end, 10_000)
                y_extern = interpFunc(x_extern)*nDensity
                basisData.append(np.array([x_extern, y_extern]))
                self.aux_indices['acetone'] = len(basisData) - 1
            else:
                print("No such external cross section: " + ef)
            
        self.basisData=basisData
        self.mols=mols
        self.isos=isos
        
    def getLinearFit(self, wl, rdts, weighted=True):
        absorptionX = wl
        absorptionY = 1.0/c_cm*(1.0/rdts- 1.0/self.vacuumRingdown(wl))
        
        if(weighted):
            weights = rdts**2
            return fitScales_weighted([absorptionX,absorptionY], self.basisData,weights)[0]
        return fitScales([absorptionX,absorptionY], self.basisData)[0]
    
    def computeTotalFitGraph(self, wl, rdts, weighted=True, vacFit = None):
        if(vacFit is not None):
            self.vac_fit=vacFit
        datStart, datEnd = np.min(wl), np.max(wl)
        datFit = self.getLinearFit(wl, rdts, weighted=weighted)
        fitGraphX, fitGraphY = getFinalModel(self.basisData,datFit, datStart, datEnd)
        #fitGraphY = self.rdtFromAlpha(fitGraphY,fitGraphX)
        return fitGraphX, fitGraphY
    
    def computeSpeciesGraphs(self, wl, rdts, weighted=True, vacFit=None, N=10000):
        from scipy.interpolate import interp1d
        if(vacFit is not None):
            self.vac_fit=vacFit
        datStart, datEnd = np.min(wl), np.max(wl)
        datFit = self.getLinearFit(wl, rdts, weighted=weighted)
        out = []
        xplot = np.linspace(datStart,datEnd,N)
        for i in range(len(self.mols)):
            interpFunc = interp1d(self.basisData[i][0], self.basisData[i][1])
            out.append([xplot, interpFunc(xplot)*datFit[i]])
        for k,v in self.aux_indices.items():
            interpFunc = interp1d(self.basisData[v][0], self.basisData[v][1])
            out.append([xplot, interpFunc(xplot)*datFit[v]])
        return out
    
    def computeFitScales(self, wl, rdts, weighted=True, vacFit = None):
        if(vacFit is not None):
            self.vac_fit=vacFit
        datFit = self.getLinearFit(wl, rdts, weighted=weighted)
        return datFit
    
    def plotFit_rdt(self, wl, rdts, weighted=True, scales = None, vacFit=None):
        if(scales is None):
            scale_fit = self.getLinearFit(wl,rdts,weighted)
        else:
            scale_fit = scales
        if(vacFit is not None):
            self.vac_fit=vacFit
            
        plots = []
        for i in range(len(self.mols)):
            b = self.basisData[i]
            lbl = self.mols[i]+"-"+str(self.isos[i])
            rdt = self.rdtFromAlpha(b[1]*scale_fit[i],b[0])
            plots.append([b[0],rdt,lbl])
        for k in self.aux_indices.keys():
            i = self.aux_indices[k]
            lbl = k
            b = self.basisData[i]
            rdt = self.rdtFromAlpha(b[1]*scale_fit[i],b[0])
            plots.append([b[0],rdt,lbl])
        return plots
