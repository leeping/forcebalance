# --------------------------------------------
# Example of how to use the RPMD integrator in
# OpenMM. 
# ---------------------------------------------
from __future__ import print_function
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout

print('------------------------------------------')
print(' PIMD simulation of q-TIP4P/F water model ')
print('------------------------------------------')

# Starting configuration to use.
pdb = PDBFile('qtip4pf.pdb')

# Force field to use. Here we use the qTIP4P/F model.
forcefield = ForceField('qtip4pf.xml')

# Initialize the parameters of the simulation. 
system = forcefield.createSystem(pdb.topology,nonbondedMethod=PME,nonbondedCutoff=1.0*nanometer,constraints=None,rigidWater=False)

# Intialize the RPMD integrator: Here we will perform a simulation at 300 Kelvin with 32 beads representing each particle
# using a time-step of 0.5 fs. The PILE thermostat will be used with a friction of 1 ps^-1. 
integrator = RPMDIntegrator(32,300*kelvin, 1.0/picosecond, 0.0005*picoseconds)

# Use CUDA Platform in mixed precision.
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'mixed'}

# Setup simulation
simulation = Simulation(pdb.topology, system, integrator, platform, properties)
simulation.context.setPositions(pdb.positions)

# The simulation will output the configuration every 100 steps.
simulation.reporters.append(PDBReporter('trajectory.pdb',100))

# The simulation will report the energies and temperature every 50 steps. 
# The temperature reported is that which the beads are evolving at which is 32 times higher than the physical temperature of the system. 
simulation.reporters.append(StateDataReporter(stdout, 1, time=True, 
    potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
    temperature=True, density=True))

# Print which Platfom is being used. Although RPMD works on the reference platform it is very slow.
print('Using Platform:',simulation.context.getPlatform().getName())

print('Starting simulation')

# Perform 20000 MD steps i.e. a 10 ps PIMD run.
simulation.step(20000)

print('Finished Simulation')
