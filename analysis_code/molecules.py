#!/usr/bin/env python
"""  molecules

  without argument:         list known molecules along with 'main' data
  with molecular name(s):   list all attributes of this molecule(s)
  with option -s:           -sh  sort according to hitran id number
                            -sg  sort according to geisa  id number
                            -sm  sort according to molecular mass
                            -sn  sort according to molecular name

  A dictionary of dictionaries containing molecular data along with some convenience functions.

"""
# NOTE: Some entries have been renamed: 'Omega' ---> 'VibFreq' (relative to Fortran)

####################################################################################################################################

molecules = {
 'H2O': {'hitran': 1,
         'geisa': 1,
	 'sao': 1,
	 'jpl':18,
         'isotopes': ['161', '181', '171', '162'],
         'mass':    18.0150, 
         'TempExpQR':    1.50000,
         'TempExpGL':   0.500000, 
         'VibFreq': [    3657.00,    1595.00,    3756.00],
         'NumDeg': [  1,  1,  1]
 },
 
 'CO2': {'hitran': 2,
         'geisa': 2,
         'sao': 2,
         'isotopes': ['626', '636', '628', '627', '638', '637', '828', '728'], 
         'mass':    44.0100, 
         'TempExpQR':    1.00000, 
         'TempExpGL':   0.750000, 
         'VibFreq': [    1388.00,    667.000,    2349.00],
         'NumDeg': [  1,  2,  1]
 },
 
'O3':   {'hitran': 3,
         'geisa': 3,
         'sao': 3,
         'jpl': 48,
         'isotopes': ['666', '668', '686'],
         'mass':    47.9980, 
         'TempExpQR':    1.5000, 
         'TempExpGL':   0.750000, 
         'VibFreq': [    1103.00,    701.000,    1042.00],
         'NumDeg': [  1,  1,  1]
 },
 
'N2O': {'hitran': 4,
        'geisa': 4,
	'sao': 4,
	'jpl': 44,
        'isotopes': ['446', '456', '546', '448', '447'],
        'mass':    44.0100, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.750000, 
        'VibFreq': [    2224.00,    589.000,    1285.00],
        'NumDeg': [  1,  2,  1]
 },
 
'CO': {'hitran': 5,
       'geisa': 5,
       'sao': 5,
       'isotopes': ['26', '36', '28', '27', '38'],
       'mass':    28.0110, 
       'TempExpQR':    1.00, 
       'TempExpGL':   0.750000, 
       'VibFreq': [    2143.00],
       'NumDeg': [  1]
 },
 
'CH4': {'hitran': 6,
        'geisa': 6,
        'sao': 6,
        'isotopes': ['211', '311', '212'],
        'mass':    16.0430, 
        'TempExpQR':    1.50, 
        'TempExpGL':   0.750000, 
        'VibFreq': [    2917.00,    1533.00,    3019.00,    1311.00],
        'NumDeg': [  1,  2,  3,  3  ]
 },
'CH3D': {'geisa': 23,
         'mass':    17.0430, 
         'isotopes': ['212', '312'],
         'TempExpQR':    1.50, 
         'TempExpGL':   0.750000, 
         'VibFreq': [    2917.00,    1533.00,    3019.00,    1311.00],
         'NumDeg': [  1,  2,  3,  3  ]
 },
 
'O2': {'hitran': 7,
       'geisa': 7,
       'sao': 7,
       'jpl': 32,
       'isotopes': ['66', '68', '67'],
       'mass':    31.9990, 
       'TempExpQR':    1.00, 
       'TempExpGL':   0.500000, 
       'VibFreq': [    1556.00],
       'NumDeg': [  1],
 },
 
'NO': {'hitran': 8,
       'geisa': 8,
       'sao': 8,
       'isotopes': ['46', '56',  '48'],
       'mass':    30.0100, 
       'TempExpQR':    1.22450, 
       'TempExpGL':   0.500000, 
       'VibFreq': [    1876.00],
       'NumDeg': [  1]
 },
 
'SO2': {'hitran':9,
        'geisa': 9,
        'sao': 9,
        'isotopes': ['626', '646'],
        'mass':    64.0600, 
        'TempExpQR':    1.50, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    1152.00,    518.000,    1362.00],
        'NumDeg': [  1,  1,  1 ] 
 },
 
'NO2': {'hitran':10,
        'geisa':10,
        'sao':10,
        'isotopes': ['646'],
        'mass':    46.0100, 
        'TempExpQR':    1.50, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    1320.00,    750.000,    1617.00],
        'NumDeg': [  1,  1,  1]
 },
 
'NH3': {'hitran':11,
        'geisa':11,
        'sao':11,
        'isotopes': ['4111', '5111'],
        'mass':    17.0300, 
        'TempExpQR':    1.50000, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    3337.00,    950.000,    3444.00,    1630.00],
        'NumDeg': [  1,  1,  1,  2]
 },
 
'HNO3':  {'hitran':12,
          'geisa':13,
          'sao':12,
          'jpl': 63,
          'isotopes': ['146'],
          'mass':    63.0100, 
          'TempExpQR':    1.50, 
          'TempExpGL':   0.500000, 
          'VibFreq': [ 3350.00, 1710.00,  1326.00,  1304.00, 879.00,  647.00,   580.00,   763.000,  458.000], 
          'NumDeg': [  1,  1,  1,  1, 1,  1,  1,  1,  1]
 },
 
'OH':  {'hitran':13,
        'geisa':14,
        'sao':13,
        'isotopes': ['61', '81', '62'],
        'mass':    17.0000, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    3570.00],
        'NumDeg': [  1]
 },
 
'HF':  {'hitran':14,
        'geisa':15,
        'sao':14,
        'isotopes': ['19'],
        'mass':    20.0100, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    3961.00],
        'NumDeg': [  1 ]
 },
 
'HCl': {'hitran':15,
        'geisa':16,
        'sao':15,
        'isotopes': ['15', '17'],
        'mass':    36.4600, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.750000, 
        'VibFreq': [    2886.00],
        'NumDeg': [  1]
 },
 
'HBr': {'hitran':16,
        'geisa':17,
        'sao':16,
        'isotopes': ['19', '11'],
        'mass':    80.9200, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    2559.00],
        'NumDeg': [  1]
 },
 
'HI':  {'hitran':17,
        'geisa':18,
        'sao':17,
        'isotopes': ['17'],
        'mass':    127.910, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    2230.00],
        'NumDeg': [  1]
 },
 
'ClO': {'hitran':18,
        'geisa':19,
        'sao':18,
        'jpl': 52,
        'isotopes': ['56', '76'],
        'mass':    51.4500, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    842.000],
        'NumDeg': [  1]
 },
 
'OCS':  {'hitran':19,
         'geisa':20,
         'sao':19,
         'isotopes': ['622', '624', '632', '822'], 
         'mass':    60.0800, 
         'TempExpQR':    1.00, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    859.000,    520.000,    2062.00],
         'NumDeg': [  1,  2,  1]
 },
 
'H2CO': {'hitran':20,
         'geisa':21,
         'sao':20,
         'isotopes': ['126', '136', '128'],
         'mass':    30.0300, 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.500000, 
         'VibFreq': [  2782.0,  1746.0,  1500.0,  1167.0,  2843.0,  1249.0],
         'NumDeg': [  1,  1,  1,  1,  1,  1  ]
 },
 
'HOCl': {'hitran':21,
         'geisa':32,
         'sao':21,
         'isotopes': ['165', '167'],
         'mass':    52.4600, 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    3609.00,    1239.00,    724.000],
         'NumDeg': [  1,  1,  1]
 },
 
'N2':  {'hitran':22,
        'geisa':33,
        'sao':22,
        'isotopes': ['44'],
        'mass':    28.0140, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    2330.00],
        'NumDeg': [  1]
 },
 
'HCN': {'hitran':23,
        'geisa':27,
        'sao':23,
        'isotopes': ['124', '134', '125'],
        'mass':    27.0300, 
        'TempExpQR':    1.00, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    2097.00,    713.000,    3311.00],
        'NumDeg': [  1,  2,  1]
 },
 
'CH3Cl': {'hitran':24,
          'geisa':34,
          'sao':24,
          'isotopes': ['215', '217'],
          'mass':    50.4900, 
          'TempExpQR':    1.50, 
          'TempExpGL':   0.500000, 
          'VibFreq': [  2968.0,  1355.0,  733.00,  3039.0,  1452.0,  1018.0],
          'NumDeg': [  1,  1,  1,  2,  2,  2  ]
 },
 
'H2O2': {'hitran':25,
         'geisa':35,
         'sao':25,
         'isotopes': ['1661'],
         'mass':    34.0100, 
         'TempExpQR':    2.00, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    3607.00,  1394.00,   863.00,  3608.00,   1266.00],
         'NumDeg': [  1,  1,  1,  1,  1]
 },
 
'C2H2': {'hitran':26,
         'geisa':24,
         'sao':26,
         'isotopes': ['1221', '1231'],
         'mass':    26.0300, 
         'TempExpQR':    1.00, 
         'TempExpGL':   0.750000, 
         'VibFreq': [    3374.00,  1974.00,  3289.00,  629.00,  730.00],
         'NumDeg': [  1,  1,  1,  2,  2]
 },
 
'C2H6': {'hitran':27,
         'geisa':22,
         'sao':27,
         'isotopes': ['1221'],
         'mass':    30.0700, 
         'TempExpQR':    1.90, 
         'TempExpGL':   0.750000, 
         'VibFreq': [  2899.00,  1375.00,  993.000,  275.000, 2954.00,  1379.00, 2994.00, 1486.00, 822.000], 
         'NumDeg': [  1,  1,  1,  1,  1,  1,  2,  1,  2]
 },
 
'PH3': {'hitran':28,
         'geisa':12,
         'sao':28,
         'isotopes': ['1111'],
         'mass':    34.0000, 
         'TempExpQR':    1.00, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    2327.00,    992.000,    1118.00,    2421.00],
         'NumDeg': [  1,  1,  2,  2 ]
 },
 
'COF2': {'hitran':29,
         'geisa':38,
         'isotopes': ['269'],
         'mass':    66.0000, 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    1944.0,  963.00,  582.00,  1242.0,  619.00,  774.00],
         'NumDeg': [  1,  1,  1,  1, 1,  1]
 },
 
'SF6': {'hitran':30,
         'geisa':39,
         'isotopes': ['29'],
         'mass':    146.000, 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.647000, 
         'VibFreq': [  774.00,  642.00,  948.00,  615.00,  523.00,  346.00],
         'NumDeg': [  1,  2,  3,  3,  3,  3 ] 
 },
 
'H2S': {'hitran':31,
         'geisa':36,
         'isotopes': ['121', '141', '131'],
         'mass':    34.0000, 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    2615.00,    1183.00,    2626.00],
         'NumDeg': [  1,  1,  1]
 },
 
'HCOOH': {'hitran':32,
          'geisa':37,
          'isotopes': ['126'],
          'mass':    46.00548, 
          'TempExpQR':    1.50, 
          'TempExpGL':   0.500000, 
          'VibFreq': [ 3570.00,  2943.00,  1770.00,  1387.00, 1229.00,  1105.00,  625.000,  1033.00,  638.000], 
          'NumDeg': [  1,  1,  1,  1, 1,  1,  1,  1,  1]
 },
 
'HO2': {'hitran':33,
        'geisa':41,
        'sao':32,
        'isotopes': ['166'],
        'mass':    33.0000, 
        'TempExpQR':    1.50, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    3436.00,    1392.00,    1098.00],
        'NumDeg': [  1,  1,  1]
 },

'ClOOCl': {'mass':    102.0000, 
           'TempExpQR':    1.50, 
           'TempExpGL':   0.500000, 
           'VibFreq': [   127.0, 328.0, 443.0, 560.0, 653.0, 752.0],
           'NumDeg': [  1, 1, 1]
 },
 
'O':      {'hitran':34,
           'mass':    16.0
 },
 
'ClONO2': {'hitran':35,
           'geisa':42,
           'mass':    97.5000, 
           'TempExpQR':    1.50, 
           'TempExpGL':   0.500000, 
           'VibFreq': [ 1735, 1292, 809, 780, 560, 434, 270, 711],
           'NumDeg': [    1,    1,   1,   1,   1,   1,   1,   1]
 },
 
'NO+': {'hitran': 36,
        'geisa': 45,
        'isotopes': ['46', '56',  '48'],
        'mass':    30.0100, 
        'TempExpQR':    1.22450, 
        'TempExpGL':   0.500000, 
        'VibFreq': [    1876.00],
        'NumDeg': [  1]
 },

'HOBr': {'hitran':37,
         'isotopes': ['169', '161'],
         'mass':    97.0000, 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    3609.00,    1239.00,    724.000],
         'NumDeg': [  1,  1,  1]
 },
  
'C2H4': {'hitran': 38,
         'geisa': 25,
         'isotopes': ['221', '311'],
         'mass':    28., 
         'TempExpQR':    1.50, 
         'TempExpGL':   0.50000, 
         'VibFreq': [ 3026, 1625, 1342, 1023, 3103, 1236, 949, 943, 3106, 826, 2989, 1444],
         'NumDeg': [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
 },
 
'CH3OH': {'hitran': 39,
          'geisa': 44,
          'mass':    32.0000, 
          'TempExpQR':    1.50, 
          'TempExpGL':   0.500000, 
          'VibFreq': [    99999.0],
          'NumDeg': [  0]
 },
 
'C3H4': {'geisa': 40,
          'mass':    40.0000, 
 },
 
'CH3Br': {'hitran': 40,
          'geisa': 43,
          'isotopes': ['219','211'],
          'mass':    94., 
 },
 
'CH3CN': {'hitran': 41,
          'geisa': 50,
         'isotopes': ['2124'],
         'mass':    41., 
 },
 
'CF4': {'hitran': 42,
        'geisa': 49,
        'isotopes': ['29'],
        'mass':    31., 
 },
 
'C3H8': {'geisa': 28,
         'isotopes': ['221'],
         'mass':    44., 
 },
 
'C2N2': {'geisa': 29,
         'isotopes': ['224'],
         'mass':    52., 
 },
 
'C4H2': {'geisa': 30,
         'isotopes': ['221'],
         'mass':    50., 
 },
 
'HC3N': {'geisa': 31,
         'isotopes': ['124'],
         'mass':    51., 
 },
 
'HNC': {'geisa': 46,
         'isotopes': ['142'],
         'mass':    27., 
 },
 
'C6H6': {'geisa': 47,
         'isotopes': ['266'],
         'mass':    78., 
 },
 
'C2HD': {'geisa': 48,
         'isotopes': ['122'],
         'mass':    17., 
 },
 
'BrO':  {
         'isotopes': ['69', '61'],
         'mass':    95.0000, 
         'TempExpQR':    1.00, 
         'TempExpGL':   0.500000, 
         'VibFreq': [    500.0],
         'NumDeg': [  0]
 },
}

####################################################################################################################################