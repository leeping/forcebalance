#!/usr/bin/env python
from forcebalance.molecule import *
from forcebalance.nifty import _exec

#Run calculation
_exec("psi4 -n 8 eth.psi4in eth.psi4out")

#Get ouptut and write qdata.txt file
mol_out = Molecule("eth.psi4out")
mol_out.write("qdata.txt", ftype="qdata")

