#!/usr/bin/env python

"""
md_one
========

This script is a part of ForceBalance and runs a single simulation
that may be combined with others to calculate general thermodynamic
properties.

This script is meant to be launched automatically by ForceBalance.

"""

#==================#
#| Global Imports |#
#==================#

import os, sys, re
import argparse
import numpy as np
import importlib as il

from forcebalance.nifty import click
from forcebalance.nifty import lp_dump, lp_load, wopen
from forcebalance.nifty import printcool, printcool_dictionary
from forcebalance.molecule import Molecule
from forcebalance.thermo import energy_derivatives

from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

#========================================================#
#| Global, user-tunable variables (simulation settings) |#
#========================================================#

# Note: Only the simulation settings that vary across different
# simulations in a target may be specified on the command line.
parser = argparse.ArgumentParser()
parser.add_argument('-T', '--temp', '--temperature', dest='temperature', type=float, 
                    help='Simulation temperature, leave blank for constant energy')
parser.add_argument('-P', '--pres', '--pressure', dest='pressure', type=float, 
                    help='Simulation pressure, leave blank for constant volume')
parser.add_argument('-g', '--grad', '--gradient', dest='gradient', action='store_true',
                    help='Calculate gradients for output time series')

# These settings may be specified for debugging purposes (i.e. they
# will override what we read from simulation.p)
parser.add_argument('-eq', '--nequil', dest='nequil', type=int, 
                    help='Number of steps for equilibration run (leave blank to use default from simulation.p)')
parser.add_argument('-md', '--nsteps', dest='nsteps', type=int, 
                    help='Number of steps for production run (leave blank to use default from simulation.p)')
parser.add_argument('-dt', '--timestep', dest='timestep', type=float, 
                    help='Time step in femtoseconds (leave blank to use default from simulation.p)')
parser.add_argument('-sp', '--sample', dest='sample', type=float, 
                    help='Sampling interval in picoseconds (leave blank to use default from simulation.p)')
parser.add_argument('-nt', '--threads', dest='threads', type=int, 
                    help='Number of threads for running parallel simulations (GMX, TINKER)')
parser.add_argument('-min', '--minimize', dest='minimize', action='store_true',
                    help='Whether to minimize the energy before starting the simulation')
parser.add_argument('-o', '-out', '--output', dest='output', type=str, nargs='+', 
                    help='Specify the time series which are written to disk')

# Parse the command line options and save as a dictionary (don't save NoneTypes)
parsed = parser.parse_args()
args = OrderedDict([(i, j) for i, j in vars(parsed).items() if j is not None])

#----
# Load the ForceBalance pickle file which contains:
#----
# - Force field object
# - Optimization parameters
# - Options loaded from file
FF, mvals = lp_load('forcefield.p')
#----
# Load the simulation pickle file which contains:
#----
# - Target options
# - Engine options
# - MD simulation options
TgtOpts, EngOpts, MDOpts = lp_load('simulation.p')
FF.ffdir = '.'

# Engine name.
engname = TgtOpts['engname']

# Import modules and create the correct Engine object.
if engname == "openmm":
    try:
        from simtk.unit import *
        from simtk.openmm import *
        from simtk.openmm.app import *
    except:
        traceback.print_exc()
        raise Exception("Cannot import OpenMM modules")
    from forcebalance.openmmio import *
    EngineClass = OpenMM
elif engname == "gromacs" or engname == "gmx":
    from forcebalance.gmxio import *
    EngineClass = GMX
elif engname == "tinker":
    from forcebalance.tinkerio import *
    EngineClass = TINKER
else:
    raise Exception('OpenMM, GROMACS, and TINKER are supported at this time.')

def main():
    
    """Usage:
    
    (prefix.sh) md_one.py -T, --temperature <temperature in kelvin>
                          -P, --pressure <pressure in atm>
                          -g, --grad (if gradients of output timeseries are desired)
                          -eq, --nequil <number of equilibration MD steps>
                          -md, --nsteps <number of production MD steps>
                          -dt, --timestep <number of production MD steps>
                          -sp, --sample <number of production MD steps>
                          -nt, --threads <number of CPU threads to use>
                          -min, --minimize <minimize the energy>
        
    This program is meant to be called automatically by ForceBalance because 
    force field options are loaded from the 'forcefield.p' file, and 
    simulation options are loaded from the 'simulation.p' file.  
    The files are separated because the same force field file
    may be used for many simulations.
    
    """

    # Write the force field file.
    FF.make(mvals)

    # Read the command line options (they may override the options from file.)
    AGrad = args['gradient']
    for i in ['temperature', 'pressure', 'nequil', 'nsteps', 'timestep', 'sample', 'threads', 'minimize']:
        if i in args:
            MDOpts[i] = args[i]
    MDOpts['nsave'] = int(1000.0*MDOpts['sample']/MDOpts['timestep'])
    if 'save_traj' in TgtOpts:
        MDOpts['save_traj'] = TgtOpts['save_traj']

    #----
    # Print some options.
    # At this point, engine and MD options should be SET!
    #----
    printcool("ForceBalance simulation using engine: %s" % engname.upper(),
              color=4, bold=True)
    printcool_dictionary(args, title="Options from command line")
    printcool_dictionary(EngOpts, title="Engine options")
    printcool_dictionary(MDOpts, title="Molecular dynamics options")

    #----
    # For convenience, assign some local variables.
    #----
    # Finite difference step size
    h = TgtOpts['h']
    # Active parameters to differentiate
    pgrad = TgtOpts['pgrad']
    # Create instances of the MD Engine objects.
    Engine = EngineClass(**EngOpts)
    click() # Start timer.
    # This line runs the condensed phase simulation.
    #----
    # The molecular dynamics simulation returns a dictionary of properties
    # In the future, the properties will be stored as data inside the object
    Results = Engine.molecular_dynamics(**MDOpts)
    if AGrad:
        Results['Potential_Derivatives'] = energy_derivatives(Engine, FF, mvals, h, pgrad, dipole=False)['potential']
    # Set up engine and calculate the potential in the other phase.
    EngOpts_ = deepcopy(EngOpts)
    EngOpts_['implicit_solvent'] = not EngOpts['implicit_solvent']
    Engine_ = EngineClass(**EngOpts_)
    Engine_.xyz_omms = Engine.xyz_omms
    Energy_ = Engine_.energy()
    Results_ = {'Potentials' : Energy_}
    if AGrad:
        Derivs_ = energy_derivatives(Engine_, FF, mvals, h, pgrad, dipole=False)['potential']
        Results_['Potential_Derivatives'] = Derivs_
    # Calculate the hydration energy of each snapshot and its parametric derivatives.
    if EngOpts['implicit_solvent']:
        Energy_liq = Results['Potentials']
        Energy_gas = Results_['Potentials']
        if AGrad: 
            Derivs_liq = Results['Potential_Derivatives']
            Derivs_gas = Results_['Potential_Derivatives']
    else:  
        Energy_gas = Results['Potentials']
        Energy_liq = Results_['Potentials']
        if AGrad: 
            Derivs_gas = Results['Potential_Derivatives']
            Derivs_liq = Results_['Potential_Derivatives']
    Results['Hydration'] = Energy_liq - Energy_gas
    if AGrad:
        Results['Hydration_Derivatives'] = Derivs_liq - Derivs_gas
    # Code of the future!
    # Don't know how to use it yet though.
    # Engine.molecular_dynamics(**MDOpts)
    # logger.info("MD simulation took %.3f seconds\n" % click())
    # # Extract properties.
    # Results = Engine.md_extract(OrderedDict([(i, {}) for i in Tgt.timeseries.keys()]))
    # potential = properties['Potential']
    # Calculate energy and dipole derivatives if needed.
    # if AGrad:
    #     Results['derivatives'] = energy_derivatives(Engine, FF, mvals, h, pgrad, dipole='dipole' in Tgt.timeseries.keys())
    # Dump results to file
    logger.info("Writing final force field.\n")
    pvals = FF.make(mvals)
    logger.info("Writing all simulation data to disk.\n")
    lp_dump(Results, 'md_result.p')
    
if __name__ == "__main__":
    main()

