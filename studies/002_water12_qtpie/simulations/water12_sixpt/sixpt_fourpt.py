#!/usr/bin/env python

from leeping.molecule import Molecule
from sys import argv
from numpy import delete

def surgery(m,excise):
    m.elem = delete(m.elem,excise)
    m.resid = delete(m.resid,excise)
    m.resname = delete(m.resname,excise)
    m.atomname = delete(m.atomname,excise)
    m.atomname = list(m.atomname)
    for i, xyz in enumerate(m.xyzs):
        m.xyzs[i] = delete(xyz,excise,axis=0)
        
M = Molecule(argv[1])

surgery(M,sum([[i*6+4, i*6+5] for i in range(12)],[]))
M.atomname = list(M.atomname)
for i in range(12):
    M.atomname[i*4] = 'OW'
    M.atomname[i*4+1] = "HW1"
    M.atomname[i*4+2] = "HW2"
    M.atomname[i*4+3] = 'MW'

print M.atomname

M.write('phixed.gro')
            
