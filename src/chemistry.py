from collections import defaultdict, OrderedDict
import re
import numpy as np

# To look up a 2-tuple of (bond energy in kJ/mol / bond order in Angstrom):
# Do BondEnergies[Elem1][Elem2][BO]
BondEnergies = defaultdict(lambda:defaultdict(dict))

## Covalent radii from Cordero et al. 'Covalent radii revisited' Dalton Transactions 2008, 2832-2838.
Radii = [0.31, 0.28, # H and He
         1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58, # First row elements
         1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, # Second row elements
         2.03, 1.76, 1.70, 1.60, 1.53, 1.39, 1.61, 1.52, 1.50, 
         1.24, 1.32, 1.22, 1.22, 1.20, 1.19, 1.20, 1.20, 1.16, # Third row elements, K through Kr
         2.20, 1.95, 1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 
         1.39, 1.45, 1.44, 1.42, 1.39, 1.39, 1.38, 1.39, 1.40, # Fourth row elements, Rb through Xe
         2.44, 2.15, 2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 
         1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87, # Fifth row elements, s and f blocks
         1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 
         1.36, 1.32, 1.45, 1.46, 1.48, 1.40, 1.50, 1.50, # Fifth row elements, d and p blocks
         2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69] # Sixth row elements

# The periodic table

PeriodicTable = OrderedDict([('H',1.0079),('He',4.0026),
                             ('Li',6.941),('Be',9.0122),('B',10.811),('C',12.0107),('N',14.0067),('O',15.9994),('F',18.9984),('Ne',20.1797),
                             ('Na',22.9897),('Mg',24.305),('Al',26.9815),('Si',28.0855),('P',30.9738),('S',32.065),('Cl',35.453),('Ar',39.948),
                             ('K',39.0983),('Ca',40.078),('Sc',44.9559),('Ti',47.867),('V',50.9415),('Cr',51.9961),('Mn',54.938),('Fe',55.845),('Co',58.9332),
                             ('Ni',58.6934),('Cu',63.546),('Zn',65.39),('Ga',69.723),('Ge',72.64),('As',74.9216),('Se',78.96),('Br',79.904),('Kr',83.8),
                             ('Rb',85.4678),('Sr',87.62),('Y',88.9059),('Zr',91.224),('Nb',92.9064),('Mo',95.94),('Tc',98),('Ru',101.07),('Rh',102.9055),
                             ('Pd',106.42),('Ag',107.8682),('Cd',112.411),('In',114.818),('Sn',118.71),('Sb',121.76),('Te',127.6),('I',126.9045),('Xe',131.293),
                             ('Cs',132.9055),('Ba',137.327),('La',138.9055),('Ce',140.116),('Pr',140.9077),('Nd',144.24),('Pm',145),('Sm',150.36),
                             ('Eu',151.964),('Gd',157.25),('Tb',158.9253),('Dy',162.5),('Ho',164.9303),('Er',167.259),('Tm',168.9342),('Yb',173.04),
                             ('Lu',174.967),('Hf',178.49),('Ta',180.9479),('W',183.84),('Re',186.207),('Os',190.23),('Ir',192.217),('Pt',195.078),
                             ('Au',196.9665),('Hg',200.59),('Tl',204.3833),('Pb',207.2),('Bi',208.9804),('Po',209),('At',210),('Rn',222),
                             ('Fr',223),('Ra',226),('Ac',227),('Th',232.0381),('Pa',231.0359),('U',238.0289),('Np',237),('Pu',244),
                             ('Am',243),('Cm',247),('Bk',247),('Cf',251),('Es',252),('Fm',257),('Md',258),('No',259),
                             ('Lr',262),('Rf',261),('Db',262),('Sg',266),('Bh',264),('Hs',277),('Mt',268)])

Elements = ["None",'H','He',
            'Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar',
            'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
            'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
            'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb',
            'Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
            'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr','Rf','Db','Sg','Bh','Hs','Mt']

BondChars = ['-','=','3']

data_from_web= """H-H 	432 	74
H-B 	389 	119
H-C 	411 	109
H-Si 	318 	148
H-Ge 	288 	153
H-Sn 	251 	170
H-N 	386 	101
H-P 	322 	144
H-As 	247 	152
H-O 	459 	96
H-S 	363 	134
H-Se 	276 	146
H-Te 	238 	170
H-F 	565 	92
H-Cl 	428 	127
H-Br 	362 	141
H-I 	295 	161
B-Cl 	456 	175
C-C 	346 	154
C=C 	602 	134
C3C 	835 	120
C-Si 	318 	185
C-Ge 	238 	195
C-Sn 	192 	216
C-Pb 	130 	230
C-N 	305 	147
C=N 	615 	129
C3N 	887 	116
C-P 	264 	184
C-O 	358 	143
C=O 	799 	120
C3O 	1072 	113
C-S 	272 	182
C=S 	573 	160
C-F 	485 	135
C-Cl 	327 	177
C-Br 	285 	194
C-I 	213 	214
Si-Si 	222 	233
Si-O 	452 	163
Si-S 	293 	200
Si-F 	565 	160
Si-Cl 	381 	202
Si-Br 	310 	215
Si-I 	234 	243
Ge-Ge 	188 	241
Ge-F 	470 	168
Ge-Cl 	349 	210
Ge-Br 	276 	230
Sn-Cl 	323 	233
Sn-Br 	273 	250
Sn-I 	205 	270
Pb-Cl 	243 	242
Pb-I 	142 	279
N-N 	167 	145
N=N 	418 	125
N3N 	942 	110
N-O 	201 	140
N=O 	607 	121
N-F 	283 	136
N-Cl 	313 	175
P-P 	201 	221
P-O 	335 	163
P=O 	544 	150
P=S 	335 	186
P-F 	490 	154
P-Cl 	326 	203
As-As 	146 	243
As-O 	301 	178
As-F 	484 	171
As-Cl 	322 	216
As-Br 	458 	233
As-I 	200 	254
Sb-Cl 	315 	232
O-O 	142 	148
O=O 	494 	121
O-F 	190 	142
S=O 	522 	143
S-S 	226 	205
S=S 	425 	149
S-F 	284 	156
S-Cl 	255 	207
Se=Se 	272 	215
F-F 	155 	142
Cl-Cl 	240 	199
Br-Br 	190 	228
I-I 	148 	267
I-F 	273 	191
I-Cl 	208 	232
Kr-F 	50 	190
Xe-O 	84 	175
Xe-F 	130 	195"""

for line in data_from_web.split('\n'):
    line = line.expandtabs()
    BE = float(line.split()[1])       # In kJ/mol
    L = float(line.split()[2]) * 0.01 # In Angstrom
    atoms = re.split('[-=3]', line.split()[0])
    A = atoms[0]
    B = atoms[1]
    bo = BondChars.index(re.findall('[-=3]', line.split()[0])[0]) + 1
    BondEnergies[A][B][bo] = (BE, L)
    BondEnergies[B][A][bo] = (BE, L)

def LookupByMass(mass):
    Deviation = 1e10
    EMatch = None
    for e, m in PeriodicTable.items():
        if np.abs(mass - m) < Deviation:
            EMatch = e
            Deviation = np.abs(mass - m)
    return EMatch

def BondStrengthByLength(A, B, length, artol = 0.33, bias=0.0): 
    # Bond length Must be in Angstrom!!
    # Set artol lower to get more aromatic bonds ; 0.5 means no aromatic bonds.
    Deviation = 1e10
    BOMatch = None
    if length < 0.5: # Assume using nanometers
        length *= 10
    if length > 50: # Assume using picometers
        length /= 100
    # A positive bias means a lower bond order.
    length += bias
    # Determine the bond order and the bond strength
    # We allow bond order 1.5 as well :)
    Devs = {}
    for BO, Vals in BondEnergies[A][B].items():
        S = Vals[0]
        L = Vals[1]
        Devs[BO] = np.abs(length-L)
        if np.abs(length-L) < Deviation:
            BOMatch = BO
            Strength = S
            Deviation = np.abs(length-L)
    if len(Devs.items()) >= 2:
        Spac = Devs[1] + Devs[2]
        Frac1 = Devs[1]/Spac
        Frac2 = Devs[2]/Spac
        if Frac1 > artol and Frac2 > artol:
            #print A, B, L, Frac1, Frac2
            BOMatch = 1.5
            Strength = 0.5 * (BondEnergies[A][B][1][0] + BondEnergies[A][B][2][0])
    return Strength, BOMatch
