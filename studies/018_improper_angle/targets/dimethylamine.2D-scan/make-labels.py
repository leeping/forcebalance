#!/usr/bin/env python

import time 
import numpy as np
import sys

# Import OpenMM tools
try:
    import openmm
    from openmm import unit
    from openmm import Platform
    from openmm.app import *
except ImportError:
    from simtk import openmm, unit
    from simtk.openmm import Platform
    from simtk.openmm.app import *

# Use MDTraj to write simulation trajectories
from mdtraj.reporters import NetCDFReporter

# Import the SMIRNOFF forcefield engine and some useful tools
from openff.toolkit.typing.engines.smirnoff import ForceField
# LPW: openff.toolkit's PME is different from openmm's PME
from openff.toolkit.typing.engines.smirnoff.forcefield import PME
from openff.toolkit.utils import get_data_filename, extractPositionsFromOEMol, generateTopologyFromOEMol

# Import the OpenEye toolkit
from openeye import oechem

pdb_filename = 'mol1.pdb'
mol_filename = 'mol1.mol2'

# Define a few simulation parameters
time_step = 2*unit.femtoseconds # simulation timestep
temperature = 300*unit.kelvin # simulation temperature
friction = 1/unit.picosecond # collision rate
num_steps = 1000000 # number of steps to run
trj_freq = 1000 # number of steps per written trajectory frame
data_freq = 1000 # number of steps per written simulation statistics

# Load molecule and create pdb object
pdb = PDBFile(pdb_filename)

# Load a SMIRNOFF forcefield
#forcefield = ForceField(get_data_filename('forcefield/Frosst_AlkEthOH_parmAtFrosst.offxml'))
forcefield = ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

# Load molecule using OpenEye tools
mol = oechem.OEGraphMol()
ifs = oechem.oemolistream(mol_filename)
# LPW: I don't understand the meaning of these lines.
# flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
# ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
oechem.OEReadMolecule(ifs, mol)
oechem.OETriposAtomNames(mol)

pdbatoms = list(pdb.topology.atoms())

labels = forcefield.labelMolecules([mol])[0]
for key, val in labels.items():
    print(key)
    for v in val:
        anames = '-'.join([pdbatoms[i].name for i in v[0]])
        anums = '-'.join([str(i) for i in v[0]])
        print("%20s %20s %5s %-s" % (anames, anums, v[1], v[2]))

# The rest of this is not needed.
sys.exit()

# Create the OpenMM system
system = forcefield.createSystem(pdb.topology, [mol], nonbondedMethod=PME, nonbondedCutoff=1.0*unit.nanometers, rigidWater=True)

# Set up an OpenMM simulation
integrator = openmm.LangevinIntegrator(temperature, friction, time_step)
platform = openmm.Platform.getPlatformByName('CUDA') 
simulation = Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(temperature)
netcdf_reporter = NetCDFReporter('water_traj.nc', trj_freq)
simulation.reporters.append(netcdf_reporter)
simulation.reporters.append(StateDataReporter('water_data.csv', data_freq, step=True, potentialEnergy=True, temperature= True, density=True))

print(simulation.context.getState(getEnergy=True).getPotentialEnergy())
simulation.minimizeEnergy()
print(simulation.context.getState(getEnergy=True).getPotentialEnergy())

# Run the simulation
print("Starting simulation")
start = time.clock()
simulation.step(num_steps)
end = time.clock()

print("Elapsed time %.2f seconds" % (end-start))
netcdf_reporter.close()
print("Done!")
