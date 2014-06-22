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
    print('\nOpenMM successfully imported\n')
except: 
    print('\nOpenMM unsuccessfully imported\n')

os.chdir(os.path.join('test','files','rpmd_files'))

ommEngine = OpenMM(ffxml='qtip4pf.xml', pdb='h2o.pdb', platname='CUDA', precision='mixed', rpmd='True')

print('Engine successfully created\n')

os.chdir('../..')

if not os.path.exists('temp'): os.mkdir('temp')

os.chdir('temp')

print('Creating simulation...\n')

# Run RPMD and save trajectory data
#ommEngine.prepare()
#ommEngine.update_simulation()
ommEngine.create_simulation(timestep=0.5, temperature=300, rpmd_opts=['32'])

print('Starting MD...\n')

out_file=open('data_output.txt','w')

data = ommEngine.molecular_dynamics(nsteps=10000, nsave=100, timestep=0.5, temperature=300, verbose=True, save_traj=True)

for elem in data:
    out_file.write(elem)

sys.exit('Finished simulation!')

