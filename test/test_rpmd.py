import unittest
import sys, os, re
import forcebalance
import abc
import numpy as np
from __init__ import ForceBalanceTestCase
from forcebalance.nifty import *
from collections import OrderedDict
from forcebalance.openmmio import OpenMM

# This script creates an OpenMM Engine object and runs a short RPMD simulation.
try:
    import simtk.openmm
    print('OpenMM successfully imported\n')
except: 
    print('OpenMM unsuccessfully imported\n')

os.chdir(os.path.join('test','files','rpmd_files'))

ommEngine = OpenMM(ffxml='qtip4pf.xml', pdb='h2o.pdb', platname='CUDA', precision='mixed')

print('Engine successfully created\n')

os.chdir('../..')

if not os.path.exists('temp'): os.mkdir('temp')

os.chdir('temp')

print('Creating simulation...\n')

# Run RPMD and save trajectory data
ommEngine.create_simulation(timestep=0.5, temperature=300, rpmd_opts=['32'])

print('Starting MD...\n')

data = ommEngine.molecular_dynamics(nsteps=10000, nsave=100, timestep=0.5, temperature=300, verbose=True, save_traj=True)

sys.exit('Finished simulation!')
