#!/usr/bin/env python

#from forcebalance.molecule import Molecule
import sys
sys.path.append("/Users/vincentvoelz/scripts/forcebalance/src")
from molecule import Molecule
import numpy as np
import os

M = Molecule('all.xyz') # These are the raw concatenated xyz files
M.atomname = [l.strip() for l in os.popen("awk '/atoms/,/bonds/ {if (/^ +[1-9]/) {print $5}}' molecule.itp").readlines()] # Get atom names from the ITP file
M.qm_energies = [float(i.split()[-1]) / 627.51 for i in M.comms] # Get QM energies from the comment line
M.qm_forces = [np.array([1.0 for j in range(3*M.na)]) for i in range(M.ns)] # Fill out forces with ones; we ignore them
# Geometries 203, 175, 173, 172 and 167 are bad.
Bad = [203, 175, 173, 172, 167]
for i in Bad:
    del M[i]
M.write('all.gro') # I prefer to use .gro format for coordinates
M.write('qdata.txt')
