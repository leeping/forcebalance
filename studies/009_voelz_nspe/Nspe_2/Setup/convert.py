#!/usr/bin/env python

# LPW uncommented this
from forcebalance.molecule import Molecule
import sys
# LPW commented out these
# sys.path.append("/Users/vincentvoelz/scripts/forcebalance/src")
# from molecule import Molecule
import numpy as np
import os

M = Molecule('all.xyz') # These are the raw concatenated xyz files
M.atomname = [l.strip() for l in os.popen("awk '/atoms/,/bonds/ {if (/^ +[1-9]/) {print $5}}' molecule.itp").readlines()] # Get atom names from the ITP file
M.qm_energies = [float(i.split()[-1])/627.51 for i in M.comms] # Get QM energies from the comment line
# M.qm_forces = [np.array([1.0 for j in range(3*M.na)]) for i in range(M.ns)] # Fill out forces with ones; we ignore them
# A crude energy cutoff that I got from viewing the energies in Gnuplot.
# Energies above this value (in a.u.) are deemed to be Bad.
Energy_Cutoff = -505000./627.51
# Get the snapshot indices which have energies lower than the cutoff.
Selection = np.where(np.array(M.qm_energies) < Energy_Cutoff)[0]
M1 = M[Selection]
# Now write the output files. :)
M1.write('all.gro') # I prefer to use .gro format for coordinates
M1.write('qdata.txt')

# print M.qm_energies
# sys.exit()
# Bad = [203, 175, 173, 172, 167]
# for i in Bad:
#     del M[i]
