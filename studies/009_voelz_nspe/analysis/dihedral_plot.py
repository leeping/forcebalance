import os, sys, glob, string
import numpy as np
import matplotlib.pyplot as plt

def Vdih_type9(dihangle, phi=0, k=1.0, n=0):
    """Returns the dihedral potential V(dihangle) = k*(1+cos(n*dihangle + phi)),
    where dihangle is in degrees."""

    return k*(1.+np.cos(n*deg2rad(dihangle) + deg2rad(phi)))

def Vdih_type4(dihangle, phi=0, k=1.0, n=0):
    """Returns the dihedral potential V(dihangle) = k*(1+cos(n*dihangle + phi)),
    where dihangle is in degrees."""

    return k*(1.+np.cos(n*deg2rad(dihangle) + deg2rad(phi)))

def deg2rad(angle):
    return np.pi*angle/180.

def parse_dihedraltypes(text):

    parms = {}

    lines = text.split('\n')
    for line in lines:
        fields = line.split()
        if len(fields) > 7:
            key = string.joinfields(fields[0:4],' ')
            dihtype  = int(fields[4])
            phi, k, n = float(fields[5]), float(fields[6]), float(fields[7])
            if not parms.has_key(key):
                parms[key] = []
            parms[key].append( (dihtype, phi, k, n) )
     
    return parms

dihedraltypes = """[ dihedraltypes ]
   hc   c3    c    n    9      180.00     0.00000           2 
   hc   c3   c3    n    9        0.00     0.65084           3 
   ca   ca   c3    n    9        0.00     0.00000           0 ; PARM 6
   ca   ca   c3    n    9        0.00     0.00000           1 ; PARM 6
   ca   ca   c3    n    9        0.00     0.00000           2 ; PARM 6
   hc   c3    c    o    9        0.00     3.34720           1
   hc   c3    c    o    9      180.00     0.33472           3
    c    n   c3   c3    9        0.00     2.21752           1
    c    n   c3   c3    9      180.00     0.62760           3
    c    n   c3   c3    9      180.00     2.09200           4
    c    n   c3   ca    9        0.00     0.00000           0
    c    n   c3   h1    9        0.00     0.00000           0
   c3    n    c    o    9      180.00    10.46000           2 ; PARM 6
   c3    c    n   c3    9      180.00    10.46000           2 ; PARM 6
   c3    n   c3   h1    9        0.00     0.00000           0
   c3   ca   ca   ca    9      180.00    15.16700           2
   c3   ca   ca   ha    9      180.00    15.16700           2
   c3   c3   ca   ca    9        0.00     0.00000           0 ; PARM 6
   c3   c3   ca   ca    9        0.00     0.00000           1 ; PARM 6
   c3   c3   ca   ca    9        0.00     0.00000           2 ; PARM 6
   ca   c3   c3   hc    9        0.00     0.65084           3
   ca   ca   ca   ca    9      180.00    15.16700           2
   ca   ca   ca   ha    9      180.00    15.16700           2
   ca   ca   c3   h1    9        0.00     0.00000           0 ; PARM 6
   ca   ca   c3   h1    9        0.00     0.00000           1 ; PARM 6
   ca   ca   c3   h1    9        0.00     0.00000           2 ; PARM 6
   c3   c3    n   c3    9        0.00     0.00000           0
   c3    n   c3   ca    9        0.00     0.00000           0
   h1   c3   c3   hc    9        0.00     0.65084           3
   ha   ca   ca   ha    9      180.00    15.16700           2
    n   c3    c    o    4      180.00    43.93200           2
   ca   ca   ca   ha    4      180.00     4.60240           2
   c3   ca   ca   ca    4      180.00     4.60240           2
    c   c3    n   c3    4      180.00     4.60240           2"""



dihangles = np.arange(-270,270)
parms = parse_dihedraltypes(dihedraltypes)

print 'number of dihedral types:', len(parms.keys())

plt.figure()
panel = 1
for key in parms.keys():

    potential = np.zeros( dihangles.shape )

    for parm in parms[key]:
        dihtype, phi, k, n = parm[0], parm[1], parm[2], parm[3]
        if dihtype == 9:
            potential += Vdih_type9(dihangles, phi, k, n)
        else:
            potential += Vdih_type4(dihangles, phi, k, n)


    plt.subplot(4,6,panel)
    plt.plot(dihangles, potential)
    if dihtype == 4:
        plt.title(key+'*improper*')
    else:
        plt.title(key)
    plt.xlabel('dihedral angle (degrees)')
    plt.ylabel('V_dih (kJ/mol)')

    panel += 1

plt.show()
