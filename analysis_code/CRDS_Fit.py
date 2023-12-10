#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:03:28 2021

@author: jaymz
"""

from .molecules import molecules
from . import hitranPlotModule as hpm
from .fitFunctions import fitScales, getFinalModel
from .external_xs.acetone import acetone_nir_xs

import numpy as np
import scipy.stats.mstats
from scipy.interpolate import interp1d
import os
import hashlib
import pickle


c_cm = 299792458*100  #Speed of light in cm/sec

HITRAN_CROSS_SECTION_FOLDER = 'hitranCrossSections/'
HITRAN_EXTENT = 2 # How much to simulate HITRAN past wavelength bounds

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

        # Check if folder exists, if not create it
        has_cs_dir = False
        with os.scandir() as dir_ents:
            for entry in dir_ents:
                if(entry.is_dir() and entry.name == HITRAN_CROSS_SECTION_FOLDER.replace('/','')):
                    has_cs_dir = True
                    break
        if(not has_cs_dir):
            os.mkdir(HITRAN_CROSS_SECTION_FOLDER)
    
    def _cs_filename(self, temperature, pressure):
        name_hash = self._filename_parameter_hash(temperature, pressure)
        isComputed = False
        with os.scandir(HITRAN_CROSS_SECTION_FOLDER) as dir_ents:
            for entry in dir_ents:
                if(entry.is_file() and entry.name == name_hash):
                    isComputed = True
                    break
        if(isComputed):
            return HITRAN_CROSS_SECTION_FOLDER + name_hash
        return None

    def _filename_parameter_hash(self, temperature, pressure):
        hashData = (hpm.wl_start,
                    hpm.wl_end,
                    self.mols,
                    self.isos,
                    temperature,
                    pressure)
        m = hashlib.sha256(usedforsecurity=False)
        m.update(str(hashData).encode())
        return m.hexdigest()

    def _get_cross_sections(self, temperature, pressure, silent=True):
        # Load or compute HITRAN cross sections
        fname_hcs = self._cs_filename(temperature, pressure)
        if(fname_hcs is not None):
            with open(fname_hcs, 'rb') as f:
                if(not silent):
                    print("Using pre-computed cross sections loaded from:")
                    print(fname_hcs)
                hit_cs=pickle.load(f)
        else:
            print("Computing new cross sections for T = %d K  and P = %d Tor at %.2f nm - %.2f nm for" % (temperature, pressure*760.15, self.wl_start, self.wl_end))
            print(self.mols)
            print(self.isos)
            hit_cs = hpm.computeCrossSections(self.mols, self.isos, temperature, pressure)
            hashname = self._filename_parameter_hash(temperature, pressure)
            hashname = HITRAN_CROSS_SECTION_FOLDER + hashname
            with open(hashname, 'wb') as f:
                pickle.dump(hit_cs, f)
                print("Saved new cross sections to: " + hashname)
        return hit_cs

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
    
    def computeHitranBasis(self,mols,isos,temperature,pressure_atm, external_fit=[], fit_offset=False, fit_slope=False, fit_quad=False, fit_cubic=False, poly_center=0.0):
        self.temperature=temperature
        self.pressureAtm = pressure_atm
        
        hpm.wl_start = self.wl_start - HITRAN_EXTENT
        hpm.wl_end   = self.wl_end + HITRAN_EXTENT

        self.mols=mols
        self.isos=isos

        #crossSections = hpm.computeCrossSections(mols,isos,temperature,pressure_atm)
        # Use caching to speed this up for things we have already computed
        crossSections = self._get_cross_sections(temperature, pressure_atm, silent=False)
        
        self.aux_indices = {}
        
        nDensity = hpm.nDensity(temperature,pressure_atm)

        basisFunctions = []
        for cs in crossSections:
            basisFunctions.append(interp1d(cs[0],cs[1]*nDensity, fill_value='extrapolate'))

        # Polynomial baseline
        if(fit_offset):
            offsetBasis = interp1d([hpm.wl_start,hpm.wl_end],[1.0,1.0], fill_value='extrapolate')
            basisFunctions.append(offsetBasis)
            self.aux_indices['offset'] = len(basisFunctions) - 1
        if(fit_slope):
            x1, x2 = hpm.wl_start, hpm.wl_end
            slopeBasis = interp1d([x1, x2],
                                   [x1-poly_center, x2-poly_center], fill_value='extrapolate')
            basisFunctions.append(slopeBasis)
            self.aux_indices['slope'] = len(basisFunctions) - 1
        if(fit_quad):
            x_poly = np.linspace(self.wl_start, self.wl_end, 10_000)
            y_poly = (x_poly - poly_center)**2
            basisFunctions.append(interp1d(x_poly, y_poly, fill_value='extrapolate'))
            self.aux_indices['quadratic'] = len(basisFunctions) - 1
        if(fit_cubic):
            x_poly = np.linspace(self.wl_start, self.wl_end, 10_000)
            y_poly = (x_poly - poly_center)**3
            basisFunctions.append(interp1d(x_poly, y_poly, fill_value='extrapolate'))
            self.aux_indices['cubic'] = len(basisFunctions) - 1

        # External cross sections
        for ef in external_fit:
            if( ef in self.external_cross_sections ):
                interpFunc = self.external_cross_sections[ef]
                x_extern = np.linspace(self.wl_start, self.wl_end, 10_000)
                y_extern = interpFunc(x_extern)*nDensity
                basisFunctions.append(interp1d(x_extern, y_extern, fill_value='extrapolate'))
                self.aux_indices['acetone'] = len(basisFunctions) - 1
            else:
                print("No such external cross section: " + ef)
            
        self.basisData=basisFunctions
        
    def getLinearFit(self, wl, rdts, weighted=True):
        absorptionX = wl
        absorptionY = 1.0/c_cm*(1.0/rdts- 1.0/self.vacuumRingdown(wl))
        
        if(weighted):
            weights = np.sqrt(rdts) # Weight function sqrt(tau)
            return fitScales([absorptionX, absorptionY], self.basisData,weights=weights)
        return fitScales([absorptionX, absorptionY], self.basisData)
    
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
            out.append([xplot, self.basisData[i](xplot)*datFit[i]])
        for k,v in self.aux_indices.items():
            out.append([xplot, self.basisData[v](xplot)*datFit[v]])
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
            
        xplot = np.linspace(self.wl_start, self.wl_end, 50_000)
        plots = []
        for i in range(len(self.mols)):
            b = self.basisData[i]
            lbl = self.mols[i]+"-"+str(self.isos[i])
            rdt = self.rdtFromAlpha(b(xplot)*scale_fit[i], xplot)
            plots.append([xplot, rdt, lbl])
        for k in self.aux_indices.keys():
            i = self.aux_indices[k]
            lbl = k
            b = self.basisData[i]
            rdt = self.rdtFromAlpha(b(xplot)*scale_fit[i], xplot)
            plots.append([xplot, rdt, lbl])
        return plots

    def plotFit(self, wl, rdts, weighted=True, scales = None, vacFit=None):
        if(scales is None):
            scale_fit = self.getLinearFit(wl,rdts,weighted)
        else:
            scale_fit = scales
        if(vacFit is not None):
            self.vac_fit=vacFit

        xplot = np.linspace(self.wl_start, self.wl_end, 50_000)
        plots = []
        for i in range(len(self.mols)):
            b = self.basisData[i]
            lbl = self.mols[i]+"-"+str(self.isos[i])
            alpha = b(xplot)*scale_fit[i]
            plots.append([xplot, alpha, lbl])
        for k in self.aux_indices.keys():
            i = self.aux_indices[k]
            lbl = k
            b = self.basisData[i]
            alpha = b(xplot)*scale_fit[i]
            plots.append([xplot, alpha, lbl])
        return plots
