#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 15:22:29 2021

@author: jaymz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.mstats
import pandas as pd

from .clusterExtract import clusterAvg

c_cm = 299792458*100  #Speed of light in cm/sec

class CRDS_analysis:
    def __init__(self, data, vacData, wlCoeffs = None):
        self.dat       = data
        self.datVacuum = vacData
        
        self.wlCoeffs = wlCoeffs
        
        temps, currs =  np.array(self.dat['ldc_params'])[:,0],\
                        np.array(self.dat['ldc_params'])[:,1]
        N = len(self.dat['rdt_mean'])
        tSec = self.dat['time'][-1]
        
                        
        print("Current Scan range:\t%.1f to %.1f\tmA" % (np.min(currs),np.max(currs)))
        print("Temperature Scan range:\t%.1f to %.1f\tC\n" % (np.min(temps),np.max(temps)))
        print("Scan Points: %d\n" % N)
        print("Total Scan Time: %d hours, %d mins, %d sec" % (np.floor(tSec/3600),np.floor((tSec%3600)/60),int(tSec%60)))
        
        
        self.currs = np.array(self.dat['ldc_params'])[:,1]
        self.vacCurrs = np.array(self.datVacuum['ldc_params'])[:,1]
        
        #-------Extract measured temperature values given by LDC--------
        if('ldc_meas' in self.dat):            #Compatiblility with older versions of CRDS program
            tm = np.array(self.dat['ldc_meas'])[:,0,0]
        else:
            tm = np.array(self.dat['ldc_params'])[:,0]
        #Replace temperatures that have None values with next value
        for i in range(len(tm))[::-1]:
            if(tm[i] is None):
                #print(i)
                tm[i] = tm[i+1]
        self.temps_measured=tm
                
        #Do the same for the vacuum data
        if('ldc_meas' in self.datVacuum):       #Compatiblility with older versions of CRDS program
            tm = np.array(self.datVacuum['ldc_meas'])[:,0,0]
        else:
            tm = np.array(self.datVacuum['ldc_params'])[:,0]
        #Replace temperatures that have None values with next value
        for i in range(len(tm))[::-1]:
            if(tm[i] is None):
                #print(i)
                tm[i] = tm[i+1]
        self.vacTemps_measured = tm
        
        self.fitVacuumData()
                
            
        
    def wavelength(self,temp,curr, coeffs = None):
        if(self.wlCoeffs is None):
            print("No Wavelength Coefficients Set!!!!")
            return
        if(coeffs is None):
            c = self.wlCoeffs
        else:
            c = coeffs
        return c[0] + c[1]*curr + c[2]*temp + \
               c[3]*curr**2 + c[4]*temp**2 + c[5]*temp*curr
               
    def fitVacuumData(self):
        currs = np.array(self.datVacuum['ldc_params'])[:,1]
        wls_vac = self.wavelength(self.vacTemps_measured,currs)
        rdts_vac = self.datVacuum['rdt_mean']
        self.vacFit = scipy.stats.mstats.theilslopes(rdts_vac,wls_vac)
        
    def vacuumRingdown(self, wavelength):
        return wavelength*self.vacFit[0] + self.vacFit[1]
    
    def rdtFromAlpha(self, alpha, wavelength):
        return 1.0/(alpha*c_cm + 1.0/self.vacuumRingdown(wavelength))
    
    def getTempSlices(self):
        #Put relevant data in easy to read variable names
        temps = np.array(self.temps_measured,dtype=float)
        #temps = np.array(dat['ldc_params'])[:,0]
        currs = self.currs
        rdts  = np.array(self.dat['rdt_mean'])
        
        splitIndices = []
        splitTemps = [self.dat['ldc_params'][0][0]]
        for i in range(0,len(temps)-1):
            if(abs(temps[i] - temps[i+1]) > 0.1):
                splitIndices.append(i+1)
                splitTemps.append(self.dat['ldc_params'][i+1][0])
        
        tempSlices = []
        tempSlices.append([temps[:(splitIndices[0])], currs[:(splitIndices[0])], rdts[:(splitIndices[0])]])
        for i in range(0,len(splitIndices) - 1):
            tempSlices.append([temps[(splitIndices[i]):(splitIndices[i+1])], currs[(splitIndices[i]):(splitIndices[i+1])], rdts[(splitIndices[i]):(splitIndices[i+1])]])
        tempSlices.append([temps[(splitIndices[-1])::],currs[(splitIndices[-1])::], rdts[(splitIndices[-1])::]])
        return tempSlices,splitTemps
    
    #Get total fit graph
    def getScaledHitran(self, wlCoeffs, crds_fit_object,plot_rdt = True):
        wl = self.wavelength(self.temps_measured, self.currs, wlCoeffs)
        hitX, hitY = crds_fit_object.computeTotalFitGraph(wl,np.array(self.dat['rdt_mean']))
        return hitX, hitY
    
    
    #Global wavelength fitting
    def fitWavelengthCoefficients(self, wlCoeffs_guess, crds_fit_object, **fit_args):
        if(crds_fit_object.basisData is None):
            print("Need to compute HITRAN basis for CRDS_Fit object!!!")
        #define an error function to be minimized using scipy.optimize
        def computeResSqr2ndOrder(params):
            temps = np.array(self.temps_measured,dtype=float)
            #temps = np.array(dat['ldc_params'])[:,0]
            currs = self.currs
            rdts  = np.array(self.dat['rdt_mean'])
            
            wavelengths = params[0] + currs*params[1] + temps*params[2] + currs**2*params[3] + temps**2*params[4] + currs*temps*params[5]
            print(params)
            absorptions = 1.0/c_cm*(1.0/rdts - 1.0/self.vacuumRingdown(wavelengths))
            
            fitGraphX, fitGraphY = crds_fit_object.computeTotalFitGraph(wavelengths, rdts, vacFit=self.vacFit)
            
            fitInterp = scipy.interpolate.interp1d(fitGraphX, fitGraphY)
            rsqr = np.sum((rdts - self.rdtFromAlpha(fitInterp(wavelengths),wavelengths))**2)
            print("%.4e" % rsqr)
            return rsqr

        # Set arguments for nonlinear fitting of wavelength coefficients
        fg = fit_args.copy()
        fg.setdefault('method', 'Nelder-Mead')
        fg.setdefault('tol', 1e-8)
        fg.setdefault('options', {'maxiter': 1000, 'maxfev':10000})
        res = scipy.optimize.minimize(computeResSqr2ndOrder,wlCoeffs_guess,
                                      method = 'Nelder-Mead',
                                      tol = 1e-8,options={'maxiter': 1000, 'maxfev':10000})
        return res
    
    def fitWavelengthPiecewise(self,crds_fit_object,wlCoeffs):
        tempSlices, splitTemps = self.getTempSlices()
        def linTempCoeffs(temp):
            return [wlCoeffs[0]-wlCoeffs[4]*temp**2, 
                    wlCoeffs[1]+wlCoeffs[5]*temp, 
                    wlCoeffs[2]+2*wlCoeffs[4]*temp,
                    wlCoeffs[3]]
        def computeResSqr2ndOrder(params, sliceIndex, scaleFits = None):
            #Put relevant data in easy to read variable names
            temps = tempSlices[sliceIndex][0]
            #temps = np.array(dat['ldc_params'])[:,0]
            currs = tempSlices[sliceIndex][1]
            rdts  = tempSlices[sliceIndex][2]
            
            wavelengths = params[0] + currs*params[1] + temps*params[2] + currs**2*params[3]
            #print(params)
            absorptions = 1.0/c_cm*(1.0/rdts - 1.0/self.vacuumRingdown(wavelengths))
            
            fitGraphX, fitGraphY = crds_fit_object.computeTotalFitGraph(wavelengths, rdts, vacFit=self.vacFit)
            
            fitInterp = scipy.interpolate.interp1d(fitGraphX, fitGraphY)
            rsqr = np.sum((rdts - self.rdtFromAlpha(fitInterp(wavelengths),wavelengths))**2)
            #print("%.4e" % rsqr)
            return rsqr
        fitResults= []
        for fitIndex in range(len(tempSlices)):
            t0 = np.mean(tempSlices[fitIndex][0])
            cGuess = linTempCoeffs(t0)
            #plt.title("Initial coefficients for slice %d" % (fitIndex))
            print("Starting optimization for slice %d/%d" % (fitIndex+1, len(tempSlices)))
            print("\tInitial:"+str(cGuess))
            res = scipy.optimize.minimize(computeResSqr2ndOrder,cGuess,
                                  args=(fitIndex),method = 'Nelder-Mead',
                                  tol = 1e-8,options={'maxiter': 1000, 'maxfev':10000})
            if(not res['success']):
                print("Convergence failed for slice %d !!" % (fitIndex))
                print("\t" + res['message'])
            print("\tFinal:  " +str(res['x'].tolist()))
            fitResults.append(res['x'])
            #plotFit(res['x'],fitIndex)
            #plt.title("Final Optimization result for slice %d" %(fitIndex))
        return fitResults
    
    def clusterExtract(self,cluster_cutoff_size = 50):
        cluserAvgs = []
        maxClusters = 0
        maxI = -1
        for i in range(len(self.dat['rdt_array'])):
            #print(i)
            ca, ns = clusterAvg(self.dat['rdt_array'][i],cutoffSize=cluster_cutoff_size)
            plotPoint = {'temp' : self.temps_measured[i],
                         'curr' : self.currs[i],
                         'clust_avgs': ca,
                         'clust_n' : ns}
            cluserAvgs.append(plotPoint)
            if(maxClusters < len(ca)):
                maxClusters = len(ca)
                maxI = i
        return cluserAvgs
    
    def arrangeClusters(self, clusts, fitX, fitY, wlCoeffs=None):
        from scipy.signal import savgol_filter
        from scipy.interpolate import interp1d
        dx = (fitX[-1]-fitX[0])/(len(fitX)-1)
        hitDeriv = savgol_filter(fitY, 13, 2,deriv=1,delta=dx)
        hdInterp = interp1d(fitX,hitDeriv)
        
        arrangedClusters = []
        dMax = np.max(np.abs(hitDeriv))
        for c in clusts:
            wl = self.wavelength(c['temp'], c['curr'],wlCoeffs)
            d = hdInterp(wl)
            if(abs(d)/dMax < .18):   #For small derivatives, keep cluster with max size
                maxI = 0
                maxN = 0
                for i in range(len(c['clust_n'])):
                    ni = c['clust_n'][i]
                    if(ni > maxN):
                        maxI = i
                        maxN = ni
                arrangedClusters.append({'wl' : wl, 
                                         'clust_avgs' : [c['clust_avgs'][maxI]] })
            elif(d > 0):
                arrangedClusters.append({'wl' : wl,
                                         'clust_avgs' : c['clust_avgs'][::-1]})
            else:
                arrangedClusters.append({'wl' : wl,
                                         'clust_avgs' : c['clust_avgs']})
        return arrangedClusters
    
    def clustersToPlotArrays(self,clusters,wlCoeffs):
        xPlots = []
        yPlots = []
        maxClusters=0
        for i in range(len(clusters)):
            maxClusters = max(len(clusters[i]['clust_avgs']),maxClusters)
        for j in range(maxClusters):
            xPlot, yPlot = [], []
            for i in range(len(clusters)):
                if(j >= len(clusters[i]['clust_avgs'])):
                    continue
                wl = self.wavelength(clusters[i]['temp'],clusters[i]['curr'], coeffs = wlCoeffs)
                xPlot.append(wl)
                yPlot.append(clusters[i]['clust_avgs'][j])
            xPlots.append(xPlot)
            yPlots.append(yPlot)
        return xPlots, yPlots
    
    def plotClusters(self, clusters, wlCoeffs=None):
        xPlots,yPlots = self.clustersToPlotArrays(clusters,wlCoeffs)
        for i in range(len(xPlots)):
            plt.plot(xPlots[i],np.array(yPlots[i])*1e6,'o',markersize=2,label = 'Cluster %d' % (i+1))
        #plt.plot(wavelength(temps_measured,currs),np.array(dat['rdt_mean'])*1e6,'o',markersize=2,label = 'Mean')
        plt.ylabel("Ringdown time (us)")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
        
    # Returns a pandas dataframe which prints nicely in Jupyter notebooks
    def getConcentrationTable(self, crds_fit_object, coeffs, piecewiseFit=False, rdt_dat=None):
        if(piecewiseFit):
            tempSlices, splitTemps = self.getTempSlices()
            wl_rc = []
            rdt_rc = []
            for i in range(len(tempSlices)):
                t, c, r = tempSlices[i]
                cFit = coeffs[i]
                wl = cFit[0] + c*cFit[1] + t*cFit[2] + c**2*cFit[3]
                wl_rc = np.concatenate([wl_rc,wl])
                rdt_rc = np.concatenate([rdt_rc,r])
            wavelengths=wl_rc
            rdts=rdt_rc
        else:
            temps = np.array(self.temps_measured,dtype=float)
            #temps = np.array(dat['ldc_params'])[:,0]
            currs = self.currs
            if(rdt_dat is None):
                rdts  = np.array(self.dat['rdt_mean'])
            else:
                rdts = np.array(rdt_dat)
            wavelengths = coeffs[0] + currs*coeffs[1] + temps*coeffs[2] + currs**2*coeffs[3] + temps**2*coeffs[4] + currs*temps*coeffs[5]
            
        datFit = crds_fit_object.computeFitScales(wavelengths,rdts,vacFit=self.vacFit)
        
        fancyTable = { 'Molecule' : [],
                       'Isotopologue ID' : [],
                       'Isotopologue Name' : [],
                       'Calculated Concentration (ppm)' :[] }

        for i in range(len(crds_fit_object.mols)):
            ppmValue=datFit[i]*1e6
            mol_id = crds_fit_object.molecules[crds_fit_object.mols[i]]['hitran']
            isoName = crds_fit_object.hapi.isotopologueName(mol_id,crds_fit_object.isos[i])
            fancyTable['Molecule'].append(crds_fit_object.mols[i])
            fancyTable['Isotopologue ID'].append(crds_fit_object.isos[i])
            fancyTable['Isotopologue Name'].append(isoName)
            fancyTable['Calculated Concentration (ppm)'].append(ppmValue)

        for i, (k, v) in enumerate(crds_fit_object.aux_indices.items()):
            ppmValue=datFit[v]*1e6
            mol_id = None
            isoName = ''
            fancyTable['Molecule'].append(k)
            fancyTable['Isotopologue ID'].append(None)
            fancyTable['Isotopologue Name'].append(isoName)
            fancyTable['Calculated Concentration (ppm)'].append(ppmValue)

        return pd.DataFrame(fancyTable)
    
    #------------Plotting functions for convenience-----------
    # Plot both the measurement and vacuum baseline, along with the linear baseline fit
    def plotDataAndVacuumFit(self, legend_args=None, **plot_args):
        wls_vac = self.wavelength(self.vacTemps_measured,self.vacCurrs)
        
        #Make some x points in the data range, just so we can plot the line
        x = np.linspace(np.min(wls_vac), np.max(wls_vac))
        
        #Sort according to wavelength
        wls = self.wavelength(self.temps_measured, self.currs)
        d= sorted(zip(wls,self.dat['rdt_mean']))
        
        wls_vac = self.wavelength(self.vacTemps_measured, self.vacCurrs)
        dVac = sorted(zip(wls_vac,self.datVacuum['rdt_mean']))
        
        data= np.transpose(d)
        dataVacuum = np.transpose(dVac)
        del d
        del dVac

        dat_args = plot_args.copy()
        size=(12,4)
        if( 'figsize' in plot_args):
            size=plot_args['figsize']
            del dat_args['figsize']
        fig = plt.figure(figsize=size)
        
        plt.plot(data[0],data[1]*1e6, label = 'Filled Chamber')
        plt.plot(dataVacuum[0],dataVacuum[1]*1e6, label = 'Empty Chamber')
        plt.plot(x,self.vacuumRingdown(x)*1e6,label = 'Theil-Sen Fit')
        plt.ylabel("Ringdown time ($\mathrm{\mu}$s)")
        plt.xlabel("Wavelength (nm)")
        if(legend_args is None):
            plt.legend()
        else:
            plt.legend(**legend_args)

    def plotResidual(self, wlCoeffs, crds_fit_object, rdt_dat=None, fontsize=16, labelsize=14, legend_args=None, **plot_args):
        temps = np.array(self.temps_measured,dtype=float)
        #temps = np.array(dat['ldc_params'])[:,0]
        currs = self.currs
        if(rdt_dat is None):
            rdts  = np.array(self.dat['rdt_mean'])
        else:
            rdts = np.array(rdt_dat)


        wavelengths = wlCoeffs[0] + currs*wlCoeffs[1] + temps*wlCoeffs[2] + \
            currs**2*wlCoeffs[3] + temps**2*wlCoeffs[4] + currs*temps*wlCoeffs[5]

        absorptions = 1.0/c_cm*(1.0/rdts - 1.0/self.vacuumRingdown(wavelengths))

        fitGraphX, fitGraphY = crds_fit_object.computeTotalFitGraph(wavelengths, rdts, vacFit = self.vacFit)
        fitGraphY = self.rdtFromAlpha(fitGraphY, fitGraphX)*1e6

        dat_args = plot_args.copy()
        fit_args = plot_args.copy()
        size=(12,4)
        if( 'figsize' in plot_args):
            size=plot_args['figsize']
            del dat_args['figsize'], fit_args['figsize']
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=size)

        dat_args.setdefault('markersize', 2)
        ax1.plot(wavelengths,rdts*1e6,'o',label='Data', **dat_args)

        fit_args.setdefault('color', 'black')
        fit_args.setdefault('lw',    0.75)
        ax1.plot(fitGraphX,fitGraphY,label='Total Fit', **fit_args)

        fitInterp = scipy.interpolate.interp1d(fitGraphX, fitGraphY)
        ax2.plot(wavelengths,rdts*1e6 - fitInterp(wavelengths), **fit_args)

        xMin, xMax = np.min(wavelengths),np.max(wavelengths)
        plt.xlim(xMin,xMax)
        ax1.set_ylabel("Ringdown time ($\mathrm{\mu}$s)", size=fontsize)
        ax2.set_ylabel("Residual ($\mathrm{\mu}$s)", size=fontsize)
        ax2.set_xlabel("Wavelength (nm)", size=fontsize)
        #ax1.grid()
        #ax2.grid()

        ax2.tick_params(axis='x', which='both', labelsize=labelsize) # font size of x axis
        ax1.tick_params(axis='y', which='both', labelsize=labelsize) # font size of y axis
        ax2.tick_params(axis='y', which='both', labelsize=labelsize) # font size of y axis

        if(legend_args is None):
            ax1.legend()
        elif(legend_args != 'off'):
            ax1.legend(**legend_args)


    # Plot the scale fit which results from the given wavelength fit coefficients
    def plotWavlengthFit(self, wlCoeffs, crds_fit_object, rdt_dat=None, fontsize=16, labelsize=14, legend_args=None, spec_plot_args=None, **plot_args):
        temps = np.array(self.temps_measured,dtype=float)
        #temps = np.array(dat['ldc_params'])[:,0]
        currs = self.currs
        if(rdt_dat is None):
            rdts  = np.array(self.dat['rdt_mean'])
        else:
            rdts = np.array(rdt_dat)


        wavelengths = wlCoeffs[0] + currs*wlCoeffs[1] + temps*wlCoeffs[2] + \
            currs**2*wlCoeffs[3] + temps**2*wlCoeffs[4] + currs*temps*wlCoeffs[5]
        xMin, xMax = np.min(wavelengths),np.max(wavelengths)

        absorptions = 1.0/c_cm*(1.0/rdts - 1.0/self.vacuumRingdown(wavelengths))

        fitGraphX, fitGraphY = crds_fit_object.computeTotalFitGraph(wavelengths, rdts, vacFit = self.vacFit)
        fitGraphY = self.rdtFromAlpha(fitGraphY, fitGraphX)*1e6

        dat_args = plot_args.copy()
        fit_args = plot_args.copy()
        spec_args = plot_args.copy()
        size=(18.8,4)
        if( 'figsize' in plot_args):
            size=plot_args['figsize']
            del dat_args['figsize'], fit_args['figsize'], spec_args['figsize']
        fig = plt.figure(figsize=size)

        ax = plt.subplot(111) # Axis handle for custom legend

        dat_args.setdefault('markersize', 2)
        ax.plot(wavelengths,rdts*1e6,'o',label='Data', **dat_args)
        specPlots = crds_fit_object.plotFit_rdt(wavelengths, rdts, vacFit=self.vacFit)

        spec_args.setdefault('color')
        # HITRAN molecules
        for i in range(0,len(crds_fit_object.mols)):
            iso_id = crds_fit_object.isos[i]
            mol_id = crds_fit_object.molecules[crds_fit_object.mols[i]]['hitran']
            if(iso_id != 1):
                lbl=crds_fit_object.hapi.isotopologueName(mol_id,iso_id)
            else:
                lbl = crds_fit_object.mols[i]
            override = False
            if(spec_plot_args is not None): # Allow user to override arguments for a specific plot
                for spa in spec_plot_args:
                    if(spa[0] == '%s-%d' % (crds_fit_object.mols[i], iso_id)):
                        ax.plot(specPlots[i][0],specPlots[i][1]*1e6,label=lbl, **spa[1])
                        override=True
                        break
            if(not override): # Remainder of keyword arguments are parssed to default plot call
                ax.plot(specPlots[i][0],specPlots[i][1]*1e6,label=lbl, **spec_args)

        # Non-HITRAN Cross sections
        for k,v in crds_fit_object.aux_indices.items():
            smask = (specPlots[v][0] >= xMin) & (specPlots[v][0] <= xMax)
            ax.plot(specPlots[v][0][smask],specPlots[v][1][smask]*1e6,label=k, **spec_args)

        # Shrink current axis by 20%, make room for legend outside plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        fit_args.setdefault('color', 'black')
        fit_args.setdefault('lw',    0.75)
        ax.plot(fitGraphX,fitGraphY,label='Total Fit', **fit_args)
        plt.xlim(xMin,xMax)
        plt.ylabel("Ringdown time ($\mathrm{\mu}$s)", size=fontsize)
        plt.xlabel("Wavelength (nm)", size=fontsize)

        ax.tick_params(axis='x', which='both', labelsize=labelsize) # font size of x axis
        ax.tick_params(axis='y', which='both', labelsize=labelsize) # font size of y axis

        # Shrink current axis by 20%
        if(legend_args != 'off'):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        if(legend_args is None):
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        elif(legend_args != 'off'):
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), **legend_args)

        #plt.legend()
        
    def plotTemperatureSlices(self):
        tempSlices, splitTemps = self.getTempSlices()
        
        plt.figure(figsize=(12,4))
        for i in range(len(tempSlices)):
            ts = tempSlices[i]
            plt.plot(self.wavelength(ts[0], ts[1]), np.array(ts[2])*1e6,'o', markersize=1,label = "T = %.1f" % (splitTemps[i]))
        plt.ylabel("Ringdown time ($\mathrm{\mu}$s)")
        plt.xlabel("Wavelength (nm)")
        plt.legend()
    
    def plotSliceFit(self,params,sliceIndex,crds_fit_object):
        tempSlices, splitTemps = self.getTempSlices()
        #Put relevant data in easy to read variable names
        temps = tempSlices[sliceIndex][0]
        #temps = np.array(dat['ldc_params'])[:,0]
        currs = tempSlices[sliceIndex][1]
        rdts  = tempSlices[sliceIndex][2]
        
        wavelengths = params[0] + currs*params[1] + temps*params[2] + currs**2*params[3]
        #print(params)
        absorptions = 1.0/c_cm*(1.0/rdts - 1.0/self.vacuumRingdown(wavelengths))
        
        fitGraphX, fitGraphY = crds_fit_object.computeTotalFitGraph(wavelengths, rdts, vacFit=self.vacFit)
        
        plt.plot(wavelengths,rdts,'o',markersize=2)
        plt.plot(fitGraphX,self.rdtFromAlpha(fitGraphY,fitGraphX),c='black')
    
    def plotScaledHitran(self, wlCoeffs, crds_fit_object,plot_rdt = True):
        wl = self.wavelength(self.temps_measured, self.currs, wlCoeffs)
        hitX, hitY = crds_fit_object.computeTotalFitGraph(wl,np.array(self.dat['rdt_mean']))
        plt.plot(hitX,self.rdtFromAlpha(hitY,hitX)*1e6,label = 'Total Hitran Fit')
        
        
    def hdfFileExport(self, wlCoeffs, crds_fit_object, outputFileName,DATA_FILE_NAME,VACUUM_FILE_NAME):
        import h5py
        
        wl = self.wavelength(self.temps_measured, self.currs, wlCoeffs)
        rdts = np.array(self.dat['rdt_mean'])
        
        fitGraphX, fitGraphY = crds_fit_object.computeTotalFitGraph(wl, rdts, vacFit = self.vacFit)
        #fitGraphY = self.rdtFromAlpha(fitGraphY, fitGraphX)*1e6
        
        N_HIT_X = 10000 #Number of points to compute in HITRAN data
        specGraphs = crds_fit_object.computeSpeciesGraphs(wl, rdts, N=N_HIT_X)
        hitOutX=specGraphs[0][0]
        #Use interpolation to compute the y abosorption axis for each molecule, and append to list
        hitOutsY = []
        for i in range(len(specGraphs)):
            hitOutsY.append( specGraphs[i][1] )
        hInterp = scipy.interpolate.interp1d(fitGraphX, fitGraphY)
        hInterp_rdt = scipy.interpolate.interp1d(fitGraphX,self.rdtFromAlpha(fitGraphY,fitGraphX)*1e6)
        hitOutTotalY = hInterp(hitOutX) #Total hitran absorption is interpolated on same x axis
        
        #Put all raw data into a pandas DataFrame for storage into HDF5 file
        rawRdts = pd.DataFrame(self.dat['rdt_array'])*1e6
        
        temps, currs = np.array(self.dat['ldc_params'])[:,0], np.array(self.dat['ldc_params'])[:,1]
        times = self.dat['time']
        std = np.std(rawRdts, axis =1)
        rdts = np.mean(rawRdts, axis = 1)
        
        dat_sorted = sorted(zip(wl, rdts, temps, currs, times, std))
        wl_sorted, rdts_sorted, temps_sorted, currs_sorted, times_sorted, std_sorted = np.transpose(dat_sorted)
        
        
        datStats = pd.DataFrame( {"Mean Ringdown Time (us)" : rdts_sorted,
                                  "Standard Deviation (us)" : std_sorted,
                                  "Time (s)" : times_sorted})
        
        ldcInfo = pd.DataFrame( {'Temperature (C)' : temps_sorted,
                                 'Current (mA)' : currs_sorted})
        wls = pd.DataFrame( {'Wavelength (nm)' : wl_sorted } )
        
        #rawData_df = pd.concat([ldcInfo, wls, datStats, rawRdts], axis = 1)
        rawData_df = pd.concat([ldcInfo, wls, datStats], axis = 1)
        
        #Create HDF5 file
        with h5py.File(outputFileName,'w') as f:
            
            #----Save relevant information about data into new dataset called "Information"
            metadata = {"Pressure (Tor)" : crds_fit_object.pressureAtm*760.15,
                        "Temperature (C)" : crds_fit_object.temperature - 273.15,
                        "Data File name": DATA_FILE_NAME,
                        "Vacuum file name" : VACUUM_FILE_NAME,
                        "Wavelength fit coefficients" : wlCoeffs,
                        "Scan Time (min)" : self.dat['time'][-1]/60,
                        "Number of scan points" : len(rdts_sorted)}
            #loop through all isotopologues and store calculated concentration
            tbl=self.getConcentrationTable(crds_fit_object, wlCoeffs)
            for i in range(len(crds_fit_object.mols)):
                ppmValue = tbl['Calculated Concentration (ppm)'][i]
                storeStr = crds_fit_object.mols[i] + '-' + str(crds_fit_object.isos[i]) + 'ppm'
                metadata[storeStr] = ppmValue

            for k,v in crds_fit_object.aux_indices.items():
                ppmValue = tbl['Calculated Concentration (ppm)'][v]
                storeStr = k + ' ppm'
                metadata[storeStr] = ppmValue

            #if(FIT_OFFSET):
            #    metadata['Absorption Offset (cm^-1)'] = datFit[0][aux_indices['offset']]
            
            md_rec = pd.DataFrame(pd.Series(metadata,dtype=str)).to_records(index_dtypes = 'S32',column_dtypes = 'S128')
            f.create_dataset("Information", data = md_rec)
            #f.attrs.update(metadata)
            
            #----Store all raw data in new dataset called "Raw Measurements"
            rawData_store = rawData_df.to_records(index = False)
            f.create_dataset("Raw Measurements", data=rawData_store)
            
        
            #-----Store ringdown data in new dataset called "Ringdown Data"
            absorption = 1.0/c_cm*(1.0/(rdts_sorted*1e-6) - 1.0/self.vacuumRingdown(wl_sorted))
            rdt_dict = {'Wavelength (nm)' : wl_sorted, 
                        'Ringdown Time (us)' : rdts_sorted, 
                        'Absorption (cm^-1)' : absorption,
                        'Absorption FIT (cm^-1)' : hInterp(wl_sorted),
                        'Ringdown FIT (us)' : hInterp_rdt(wl_sorted)}
            rdt_store = pd.DataFrame(rdt_dict).to_records(index=False)
            f.create_dataset("Ringdown Data", data = rdt_store)
        
            #-----Store vacuum data in new dataset called "Vacuum Data"
            vacX = np.array(self.datVacuum['wavelength'])
            vac_dict = {"Wavelength (nm)" : vacX,
                        "Ringdown Time (us)" : np.array(self.datVacuum['rdt_mean'])*1e6,
                        "Vacuum Fit (us)" : self.vacuumRingdown(vacX)}
            vac_store = pd.DataFrame(vac_dict).to_records(index=False)
            f.create_dataset("Vacuum Data", data = vac_store)
        
            #----Store curves resulting from HITRAN fit
            names = ['Wavelength (nm)']
            hitran_dict = {'Wavelength (nm)' : hitOutX}
            for i in range(len(crds_fit_object.mols)): #Loop through all hitranPlot molecules
                dsName = crds_fit_object.mols[i] + '-' + str(crds_fit_object.isos[i]) + ' Absorption (cm^-1)'
                hitran_dict[dsName] = hitOutsY[i]
                dsName = crds_fit_object.mols[i] + '-' + str(crds_fit_object.isos[i]) + ' Ringdown (us)'
                hitran_dict[dsName] = self.rdtFromAlpha(hitOutsY[i],hitOutX)

            for k,v in crds_fit_object.aux_indices.items():
                dsName = k + ' Absorption (cm^-1)'
                hitran_dict[dsName] = hitOutsY[v]
                dsName = k + ' Ringdown (us)'
                hitran_dict[dsName] = self.rdtFromAlpha(hitOutsY[v],hitOutX)

        
            #Store total HITRAN fit
            hitran_dict['Total Fit (cm^-1)'] = hitOutTotalY
        
            #Store HITRAN absorption converted back to ringdown time
            hitran_dict["Total Fit Ringdown Time (us)"] = hInterp_rdt(hitOutX)
        
            #Store all hitran curves in dataset called "HITRAN fit"
            dsetData = pd.DataFrame(hitran_dict).to_records(index=False)
            f.create_dataset("HITRAN Fit", data = dsetData)
