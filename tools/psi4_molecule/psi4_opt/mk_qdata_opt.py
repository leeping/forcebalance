#!/usr/bin/env python
from forcebalance.molecule import *
from forcebalance.nifty import _exec

#Get output from geometry optimization
mol = Molecule("eth_opt.psi4out")
#Write outputs as xyz and qdata file
mol.write("output.xyz", ftype="xyz")

#If you want to run some other Psi4 calculation
#then you can change the settings read by Molecule
#through Molecule.psi4args and write a new calculation input
mol.psi4args["calc"] = ["energy('mp2')"]
mol.psi4args["set"]["basis"] = ["aug-cc-pVTZ"] 

#Write new calculation input from the final structure
#of the optimization
mol.write("eth_energy.psi4in", selection=-1)
_exec("psi4 -n 8 eth_energy.psi4in eth_energy.psi4out")

#Read in output and write qdata.txt file
mol_energy = Molecule("eth_energy.psi4out")
mol_energy.write("qdata.txt", ftype="qdata")

