#!/usr/bin/env python

import sys
from forcebalance.molecule import *

# Script to generate virtual sites and rename atoms in .gro file.

M = Molecule(sys.argv[1])
if 'M' in M.elem:
    print "Virtual sites already exist"
    sys.exit()
num_mol = M.na/3

for i in range(num_mol)[::-1]:
    v = i*3 + 3
    M.add_virtual_site(v, resid=i+1, elem='M', atomname='MW', resname='SOL', pos=i*3)

M.replace_peratom('resname', 'HOH','SOL')
M.replace_peratom_conditional('resname', 'SOL', 'atomname', 'H1', 'HW1')
M.replace_peratom_conditional('resname', 'SOL', 'atomname', 'H2', 'HW2')
M.replace_peratom_conditional('resname', 'SOL', 'atomname', 'O', 'OW')
M.write('new.gro')
