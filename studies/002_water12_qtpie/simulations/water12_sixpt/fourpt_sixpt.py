#!/usr/bin/env python

from leeping.molecule import Molecule
from sys import argv
from numpy import arange, array, delete, insert, vstack

def surgery(m,excise):
    m.elem = delete(m.elem,excise)
    m.resid = delete(m.resid,excise)
    m.resname = delete(m.resname,excise)
    m.atomname = delete(m.atomname,excise)
    for i, xyz in enumerate(m.xyzs):
        m.xyzs[i] = delete(xyz,excise,axis=0)

M = Molecule(argv[1])

A = vstack((arange(0,48,4)+4,arange(0,48,4)+4))
A = A.flatten('F')
B = vstack((arange(12)+1,arange(12)+1))
B = B.flatten('F')
C = vstack((array(['L1' for i in range(12)]),array(['L2' for i in range(12)])))
C = C.flatten('F')

#print A

#exit()

for i in range(12):
    M.atomname[i*4] = 'O'
    M.atomname[i*4+1] = "H1"
    M.atomname[i*4+2] = "H2"
    M.atomname[i*4+3] = 'M'

M.elem = insert(M.elem,A,'L')
M.resid = insert(M.resid,A,B)
M.resname = insert(M.resname,A,'SOL')
M.atomname = insert(M.atomname,A,C)
for i, xyz in enumerate(M.xyzs):
    M.xyzs[i] = insert(M.xyzs[i],A,array([0,0,0]),axis=0)
    
#M.elem = insert(M.elem,range(0,48,4),'L')

#def insertion(m,):

#surgery(M,sum([[i*6+4, i*6+5] for i in range(12)],[]))
## M.atomname = list(M.atomname)

print M.elem
print M.resid
print M.atomname


M.write('phixed.gro')
            
