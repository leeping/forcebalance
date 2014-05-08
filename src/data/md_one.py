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
# will override what we read from forcebalance.p)
parser.add_argument('-eq', '--nequil', dest='nequil', type=int, 
                    help='Number of steps for equilibration run (leave blank to use default from forcebalance.p)')
parser.add_argument('-md', '--nsteps', dest='nsteps', type=int, 
                    help='Number of steps for production run (leave blank to use default from forcebalance.p)')
parser.add_argument('-dt', '--timestep', dest='timestep', type=float, 
                    help='Time step in femtoseconds (leave blank to use default from forcebalance.p)')
parser.add_argument('-sp', '--sample', dest='sample', type=float, 
                    help='Sampling interval in picoseonds (leave blank to use default from forcebalance.p)')
parser.add_argument('-nt', '--threads', dest='threads', type=int, 
                    help='Sampling interval in picoseonds (leave blank to use default from forcebalance.p)')
parser.add_argument('-min', '--minimize', dest='minimize', action='store_true',
                    help='Whether to minimize the energy before starting the simulation')
parser.add_argument('-o', '-out', '--output', dest='output', type=str, nargs='+', 
                    help='Specify the time series which are written to disk')

# Parse the command line options and save as a dictionary (don't save NoneTypes)
parsed = parser.parse_args()
args = OrderedDict([(i, j) for i, j in vars(parsed).items() if j != None])

#----
# Load the ForceBalance pickle file which contains:
#----
# - Force field object
# - Optimization parameters
# - Options loaded from file
FF, mvals, Sim = lp_load(open('forcebalance.p'))
FF.ffdir = '.'

# Engine name.
engname = Sim.engname

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
        
    This program is meant to be called automatically by ForceBalance
    because most options are loaded from the 'forcebalance.p'
    simulation file.
    
    """

    # Write the force field file.
    FF.make(mvals)

    # Read the command line options (they may override the options from file.)
    AGrad = args['gradient'] or Sim.gradient
    for i in ['temperature', 'pressure', 'nequil', 'nsteps', 'timestep', 'sample', 'threads', 'minimize']:
        if i in args:
            Sim.MDOpts[i] = args[i]

    #----
    # Print some options.
    # At this point, engine and MD options should be SET!
    #----
    printcool("ForceBalance simulation using engine: %s" % engname.upper(),
              color=4, bold=True)
    printcool_dictionary(args, title="Options from command line")
    printcool_dictionary(Sim.EngOpts, title="Engine options")
    printcool_dictionary(Sim.MDOpts, title="Molecular dynamics options")

    #----
    # For convenience, assign some local variables.
    #----
    # Finite difference step size
    h = Sim.h
    # Active parameters to differentiate
    pgrad = Sim.pgrad
    # Create instances of the MD Engine objects.
    Engine = EngineClass(name=Sim.type, **Sim.EngOpts)
    click() # Start timer.
    # This line runs the condensed phase simulation.
    Engine.molecular_dynamics(**Sim.MDOpts)
    logger.info("MD simulation took %.3f seconds\n" % click())
    # Extract properties.
    Results = Engine.md_extract(OrderedDict([(i, {}) for i in Sim.timeseries.keys()]))
    # Calculate energy and dipole derivatives if needed.
    if AGrad:
        Results['derivatives'] = energy_derivatives(Engine, FF, mvals, h, pgrad, dipole='dipole' in Sim.timeseries.keys())
    # Dump results to file
    logger.info("Writing final force field.\n")
    pvals = FF.make(mvals)
    logger.info("Writing all simulation data to disk.\n")
    with wopen('md_result.p') as f:
        lp_dump(Results, f)
    
if __name__ == "__main__":
    main()

