#!/usr/bin/env python

import os, sys, re
import numpy as np
from forcebalance.molecule import Molecule
from forcebalance.readfrq import read_frq_tc

# TeraChem frequency output file.
tcout = sys.argv[1]

# Starting coordinate file.
M = Molecule(sys.argv[2])
xyz = M.xyzs[0]
M.xyzs = []
M.comms = []

# Mode number, starting from 1.
modenum = int(sys.argv[3])
if modenum == 0:
    raise RuntimeError('Mode numbers start from 1')

frqs, modes = read_frq_tc(tcout)

xmode = modes[modenum - 1].reshape(-1,3)

xmodeNorm = np.array([np.linalg.norm(i) for i in xmode])
idxMax = np.argmax(xmodeNorm)
print "In mode #%i, the largest displacement comes from atom #%i (%s); norm %.3f" % (modenum, idxMax+1, M.elem[idxMax], np.max(xmodeNorm))

xmode *= 0.3 # Reasonable vibrational amplitude

spac = np.linspace(0, 2*np.pi, 101)
disp = np.sin(spac)

for i in disp:
    M.xyzs.append(xyz+i*xmode)

M.comms = ['Vibrational Mode %i Frequency %.3f Displacement %.3f' % (modenum, frqs[modenum-1], disp[i]*(np.linalg.norm(xmode)/np.sqrt(M.na))) for i in range(len(M))]
M.write(os.path.splitext(tcout)[0]+'.mode%03i.xyz' % modenum)
