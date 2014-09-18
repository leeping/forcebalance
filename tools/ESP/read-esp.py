#!/usr/bin/env python

import numpy as np
import os, sys
from forcebalance.nifty import isfloat
from forcebalance.molecule import Molecule

def read_psi_xyzesp(psiout):
    # Read Psi4 ESP output file for geometries, ESP values and grid points.
    XMode = 0
    EMode = 0
    ESPMode = 0
    xyzs = []
    xyz = []
    elem = []
    espxyz = []
    espval = []
    for line in open(psiout):
        s = line.split()
        if XMode == 1:
            if len(s) == 4 and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
                e = s[0]
                xyz.append([float(i) for i in s[1:4]])
                if EMode == 1:
                    elem.append(e)
            elif len(xyz) > 0:
                xyzs.append(np.array(xyz))
                xyz = []
                XMode = 0
        if ESPMode == 1:
            if len(s) == 4 and isfloat(s[0]) and isfloat(s[1]) and isfloat(s[2]) and isfloat(s[3]):
                espxyz.append([float(i) for i in s[:3]])
                espval.append(float(s[3]))
            elif len(espxyz) > 0:
                # After reading in a block of ESPs, don't read any more.
                ESPMode = -1 
        if line.strip().startswith("Geometry (in Angstrom)"):
            XMode = 1
            EMode = len(elem) == 0
        if 'Electrostatic Potential' in line.strip() and ESPMode == 0:
            ESPMode = 1
    if len(xyzs) == 0:
        raise Exception('%s has length zero' % psiout)
    return xyzs, elem, espxyz, espval

xyzs, elem, espxyz, espval = read_psi_xyzesp(sys.argv[1])

M = Molecule()
M.xyzs = xyzs
M.elem = elem
M.write('%s.xyz' % os.path.splitext(sys.argv[1])[0])

EM = Molecule()
EM.xyzs = [np.array(espxyz) * 0.52917721092]
EM.elem = ['H' for i in range(len(espxyz))]
EM.write('%s.espx' % os.path.splitext(sys.argv[1])[0], ftype="xyz")

M.qm_espxyzs = EM.xyzs
M.qm_espvals = [np.array(espval)]
M.write("qdata.txt")

np.savetxt('%s.esp' % os.path.splitext(sys.argv[1])[0], espval)
