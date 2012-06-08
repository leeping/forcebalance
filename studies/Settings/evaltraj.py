#!/usr/bin/env python

"""
@package evaltraj

Trajectory evaluation of energies and (optionally) energy derivatives
in OpenMM.  Loops over a provided DCD trajectory and computes energies
at specified force field parameters.  Parameter values are provided
using the pickled ForceBalance force field and mathematical parameter
values, as in npt.py.

@author Lee-Ping Wang <leeping@stanford.edu>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness for a
particular purpose.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#================#
# Global Imports #
#================#

import os
import os.path
import sys
import math
import numpy as np
import simtk
import simtk.unit as units
import simtk.openmm as openmm
from simtk.openmm.app import *
from forcebalance.forcefield import FF
from forcebalance.nifty import col, flat, lp_dump, lp_load, printcool, printcool_dictionary
from forcebalance.finite_difference import fdwrap, f12d3p
from forcebalance.molecule import Molecule
from forcebalance.openmmio import liquid_energy_driver, liquid_energy_derivatives

#======================================================#
# Global, user-tunable variables (simulation settings) #
#======================================================#

# Name of the simulation platform (Reference, Cuda, OpenCL)
PlatName = 'Cuda'

mutual_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*units.nanometer, 
                 'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 1.2, 
                 'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                 'mutualInducedTargetEpsilon' : 1e-6}

direct_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*units.nanometer, 
                 'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 1.2, 
                 'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                 'polarization' : 'direct'}

tip3p_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*units.nanometer, 
                'vdwCutoff' : 1.2, 'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24]}

mono_kwargs = {'nonbondedMethod' : NoCutoff, 'constraints' : None, 
               'rigidWater' : False, 'polarization' : 'direct'}

simulation_settings = direct_kwargs
if 'tip3p' in sys.argv[2]:
   print "Using TIP3P settings."
   PlatName = 'OpenCL'
   simulation_settings = tip3p_kwargs

#================================#
# Create the simulation platform #
#================================#
print "Setting Platform to", PlatName
platform = openmm.Platform.getPlatformByName(PlatName)
# Set the device to the environment variable or zero otherwise
device = os.environ.get('CUDA_DEVICE',"0")
print "Setting Device to", device
platform.setPropertyDefaultValue("CudaDevice", device)
platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
          
def main():
   
   """ 
   Usage: (runcuda.sh) evaltraj.py protein.pdb forcefield.xml trajectory.dcd [yes, true, 1]

   This program is meant to be called automatically by ForceBalance on
   a GPU cluster (specifically, subroutines in openmmio.py).  It is
   not easy to use manually.  This is because the force field is read
   in from a ForceBalance 'FF' class.

   I wrote this program because automatic fitting of the density (or
   other equilibrium properties) is computationally intensive, and the
   calculations need to be distributed to the queue.  The main instance
   of ForceBalance (running on my workstation) queues up a bunch of these
   jobs (using Work Queue).  Then, I submit a bunch of workers to GPU
   clusters (e.g. Certainty, Keeneland).  The worker scripts connect to
   the main instance and receives one of these jobs.

   This script can also be executed locally, if you want to (e.g. for
   debugging).  Just make sure you have the pickled 'forcebalance.p'
   file.

   """
   
   # Create an OpenMM PDB object so we may make the Simulation class.
   pdb = PDBFile(sys.argv[1])
   # Load the force field in from the ForceBalance pickle.
   FF,mvals,h = lp_load(open('forcebalance.p'))
   # Create the force field XML files.
   FF.make(mvals)
   # Make use of my Molecule class to load in the DCD file (because OpenMM can't do this as of 4.1)
   Trajectory = Molecule(sys.argv[1])
   Trajectory.load_frames(sys.argv[3])
   # Use the OpenMM syntax
   Xyzs = Trajectory.openmm_positions()
   Boxes = Trajectory.openmm_boxes()
   # Now that we have the coordinates, we can compute the energy and its derivatives.
   Energies = liquid_energy_driver(mvals, pdb, FF, Xyzs, simulation_settings, platform, Boxes)
   # The fourth argument is used to trigger the computation of derivatives.
   if len(sys.argv) > 4 and sys.argv[4].upper() in ['1','YES','TRUE','Y','YEAH','PLEASE']:
      G, Hd = liquid_energy_derivatives(mvals, h, pdb, FF, Xyzs, simulation_settings, platform, Boxes)
   else:
      G = None
   with open(os.path.join('evaltraj_result.p'),'w') as f: lp_dump((Energies, G),f)

if __name__ == "__main__":
   main()
