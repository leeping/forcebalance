#!/usr/bin/env python

from forcebalance.molecule import Molecule
import os, sys

# Load in the Gromacs .gro file to be converted to TINKER format.
M = Molecule(sys.argv[1])

# Build the line suffix for the TINKER format.
tinkersuf = []
for i in range(M.na):
    if i%3==0:
        tinkersuf.append("%5i %5i %5i" % (1, i+2, i+3))
    else:
        tinkersuf.append("%5i %5i" % (2, i-i%3+1))
M.tinkersuf = tinkersuf

# Delete the periodic box.
del M.Data['boxes']

# Write the TINKER output format.
M.write(os.path.splitext(sys.argv[1])[0]+'.xyz', ftype='tinker')
