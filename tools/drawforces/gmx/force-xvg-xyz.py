#!/usr/bin/env python

from forcebalance.molecule import *
import numpy as np
import sys

# A simple 
# Run this script as: <script_name> gmx-all.gro gmx-f.xvg 10

# Convert gmx forces (kJ/mol/nm) to gradients in a.u.
fqcgmx = -49621.9

M = Molecule(sys.argv[1])
xvg = np.loadtxt(sys.argv[2])

# Multiply forces by a further user-specified constant to aid visualization
const = float(sys.argv[3])
 
xvgxyz = []
for i in xvg:
    xvgxyz.append(i[1:].reshape(-1,3)/fqcgmx*const)
    
M.xyzs = xvgxyz
M.write('gmx-grad.xyz')
