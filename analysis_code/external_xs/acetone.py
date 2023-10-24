# Empirical fit of acetone cross section
# from calibration gas.
# Range 1671.5nm to 1675nm

def acetone_nir_xs(wavelength_nm):
    c0 = 0.33624069085491015
    c1 = 0.7219321182176237
    c2 = 9.65931401e-06
    c3 = 1672.5648640468119
    c4 = 2.98965919e-05
    w = c0*(wavelength_nm-1672.5) + c1

    return (c2*w / (w**2 + (wavelength_nm - c3)**2) + c4) / (3.2618e16)
