#!/usr/bin/env python

import os, sys, re
import numpy as np
from forcebalance.molecule import Molecule
from forcebalance.readfrq import read_frq_gen

# Frequency output file.
fout = sys.argv[1]

# Mode number, starting from 1.
modenum = int(sys.argv[2])

if modenum == 0:
    raise RuntimeError("Start mode number from one, please")

frqs, modes, intens, elem, xyz = read_frq_gen(fout)

M = Molecule()
M.elem = elem[:]
M.xyzs = []

xmode = modes[modenum - 1]
xmode /= (np.linalg.norm(xmode)/np.sqrt(M.na))
xmode *= 0.3 # Reasonable vibrational amplitude

spac = np.linspace(0, 1, 101)
disp = np.concatenate((spac, spac[::-1][1:], -1*spac[1:], -1*spac[::-1][1:-1]))

for i in disp:
    M.xyzs.append(xyz+i*xmode.reshape(-1,3))

M.comms = ['Vibrational Mode %i Frequency %.3f Displacement %.3f' % (modenum, frqs[modenum-1], disp[i]*(np.linalg.norm(xmode)/np.sqrt(M.na))) for i in range(len(M))]

M.write(os.path.splitext(fout)[0]+'.mode%03i.xyz' % modenum)
