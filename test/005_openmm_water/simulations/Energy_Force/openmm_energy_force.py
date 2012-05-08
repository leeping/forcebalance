#!/usr/bin/env python

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import numpy as np
import sys
import pickle

def main():
    ################################################
    # OpenMM Script for Computing Energy and Force #
    ################################################
    # Takes three arguments: 
    # PDB file (for building Topology)
    # GRO file (for Coordinates in all frames)
    # Force field file
    pdb = PDBFile(sys.argv[1])
    gro = GroFile(sys.argv[2])
    forcefield = ForceField(sys.argv[3])

    ################################################
    #       Simulation settings (IMPORTANT)        #
    # Agrees with TINKER to within 0.0001 kcal! :) #
    ################################################
    # Use for Mutual
    # system = forcefield.createSystem(pdb.topology,rigidWater=False,mutualInducedTargetEpsilon=1e-6)
    # Use for Direct
    system = forcefield.createSystem(pdb.topology,rigidWater=False,polarization='Direct')

    # Create the simulation; we're not actually going to use the integrator
    integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)

    sys.stderr.write("openmm_energy_force.py: Computing the potential energy scan\n")
    sys.stderr.write("openmm_energy_force.py: Energies are in units of kJ/mol\n")
    sys.stderr.write("openmm_energy_force.py: I am using the %s platform\n" % simulation.context.getPlatform().getName())

    Energies = []
    Forces = []
    # Loop through the snapshots
    for i in range(len(gro.positions)):
        # Set the positions using the loaded GROMACS file
        simulation.context.setPositions(gro.getPositions(index=i))
        # Compute the potential energy and append to list
        Energies.append(simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole)
        # Compute the force and append to list
        Forces.append(np.array(simulation.context.getState(getForces=True).getForces() / kilojoules_per_mole * nanometer).flatten())
    Energies = np.array(Energies)
    Forces   = np.array(Forces)
    Answer   = {'Energies': Energies, 'Forces' : Forces}
    with open('Answer.dat','w') as f: pickle.dump(Answer,f)

if __name__ == "__main__":
    main()
