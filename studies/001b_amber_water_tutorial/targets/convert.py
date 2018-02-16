#!/usr/bin/env python

from forcebalance.molecule import *

M = Molecule('all.gro')
M1 = M.atom_select([i for i in range(M.na) if i%4 != 3])
M1[0].write('conf.pdb')
M1.write('all.mdcrd')
