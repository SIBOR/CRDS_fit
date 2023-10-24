from .molecules import molecules
from . import hapi

import numpy as np

HITRAN_LINE_FOLDER = 'hitranLineFiles/'

#Specify wavelength range in nm to compute HITRAN cross sections
wl_start, wl_end = 1640.0, 1660.0

#Resolution to compute HITRAN cross sections
WAVENUMBER_STEP_DEFAULT = 0.001

#Number density of ideal gas in molecules/cm^3
def nDensity(temp,pressAtm):
    return pressAtm/0.08205/temp*6.022e23/1000

def plotHitranCurves(mols, isos, temp, pressAtm, wavenumber_step=WAVENUMBER_STEP_DEFAULT):
    #Fetch the line information from the HITRAN database. 
    #This information is saved to files in this folder for each molecule and isotopologue
    for i in range(0,len(mols)):
        m = mols[i]
        iso = isos[i]
        hapi.fetch(HITRAN_LINE_FOLDER + m + '-' + str(iso), molecules[m]['hitran'], iso, 1e7/wl_end, 1e7/wl_start)
    #Calculate HITRAN cross sections for selected molecules
    hitranPlots = []
    for i in range(len(mols)):
        m = mols[i]
        iso = isos[i]
        tblStr = (HITRAN_LINE_FOLDER + m + '-' + str(iso))
        nu, coef = hapi.absorptionCoefficient_Voigt(SourceTables=tblStr,
                                                    Diluent={'air':1.0}, HITRAN_units = True, 
                                                    Environment = {'T':temp,'p':pressAtm},
                                                    WavenumberStep = wavenumber_step,WavenumberWing = 20.0)
        hitranPlots.append([1e7/nu,np.array(coef)*nDensity(temp,pressAtm),m])
    return hitranPlots

def computeCrossSections(mols, isos, temp, pressAtm, wavenumber_step=WAVENUMBER_STEP_DEFAULT):
    #Fetch the line information from the HITRAN database. 
    #This information is saved to files in this folder for each molecule and isotopologue

    lineFolder = HITRAN_LINE_FOLDER

    for i in range(0,len(mols)):
        m = mols[i]
        iso = isos[i]
        hapi.fetch(lineFolder + m + '-' + str(iso), molecules[m]['hitran'], iso, 1e7/wl_end, 1e7/wl_start)       
    #Calculate HITRAN cross sections for selected molecules
    hitranPlots = []
    for i in range(len(mols)):
        m = mols[i]
        iso = isos[i]
        tblStr = (lineFolder + m + '-' + str(iso))
        nu, coef = hapi.absorptionCoefficient_Voigt(SourceTables=tblStr,
                                                    Diluent={'air':1.0}, HITRAN_units = True, 
                                                    Environment = {'T':temp,'p':pressAtm},
                                                    WavenumberStep = wavenumber_step,WavenumberWing = 20.0)
        abund = hapi.abundance(molecules[m]['hitran'], iso)
        #Convert to wavelength in nm and undo abundance scaling
        hitranPlots.append([1e7/nu,np.array(coef)/abund,m+'-'+str(iso)])
    return hitranPlots