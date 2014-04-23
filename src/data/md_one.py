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

from forcebalance.nifty import lp_dump, lp_load, wopen
from forcebalance.nifty import printcool, printcool_dictionary
from forcebalance.molecule import Molecule

from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

#========================================================#
#| Global, user-tunable variables (simulation settings) |#
#========================================================#

# Note: Only the simulation settings that vary across different
# simulations in a target may be specified on the command line.

# ANYTHING THREE LETTERS OR BELOW IS A SHORT OPTION WITH ONE DASH!

parser = argparse.ArgumentParser()
parser.add_argument('simulation', type=str,
                    help='The simulation name (important; used in setting up)')
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

Copts = vars(parser.parse_args())

def main():
    
    """Usage:
    
    (prefix.sh) md_one.py <name of simulation>
                               -T, --temperature <temperature in kelvin>
                               -P, --pressure <pressure in atm>
                               -g, --grad (if gradients of output timeseries are desired)
                               -o, --outputs <list of output time series>
                               -eq, --nequil <number of equilibration MD steps>
                               -md, --nsteps <number of production MD steps>
                               -dt, --timestep <number of production MD steps>
                               -nt, --interval <number of production MD steps>
        
    This program is meant to be called automatically by ForceBalance
    because most options are loaded from the 'forcebalance.p' input
    file.
    
    """

    # printcool("ForceBalance simulation using engine: %s" % engname.upper(),
    #           color=4, bold=True)

    #----
    # Load the ForceBalance pickle file which contains:
    #----
    # - Force field object
    # - Optimization parameters
    # - Options loaded from file
    FF, mvals, Fopts = lp_load(open('forcebalance.p'))
    FF.ffdir = '.'
    # Write the force field file.
    FF.make(mvals)

    printcool_dictionary(Copts, title="Options from command line")
    printcool_dictionary(Fopts, title="Options from file")

    # Read the command line options (they can override the options from file.)
    # Calculate energy / dipole derivatives.
    AGrad = Copts['gradient'] or Fopts['gradient']
    # Whether to minimize the energy.
    minimize = Copts['minimize'] or Fopts['minimize']
    # Engine name.
    engname = Fopts['engname']
    # 
    threads = Copts.get('threads', Fopts.get('threads', 1))

    # # Get the temperature.
    # temperature = Copts.get('temperature', Fopts.get('temperature', None))
    # # Get the pressure.
    # pressure = Copts.get('pressure', Fopts.get('pressure', None))
    # # 
    # nequil = Copts.get('nequil', Fopts.get('nequil'))
    
    #----
    # load some options from file
    #----
    # Finite difference step size
    h = Fopts['h']
    # Active parameters for taking the gradient
    pgrad = Fopts['pgrad']
    # Name of the initial coordinate file
    coords = Fopts['coords']
    # Base name of the initial coordinate file
    cbase = os.path.splitext(coords)[0]
    # Actually start to do stuff.
    # Molecule object corresponding to 
    M = Molecule(coords)

    #----
    # Engine options
    #----
    EngOpts = OrderedDict([("coords", coords), ("pbc", Fopts['pbc'])])
    if engname == "openmm":
        if pbc:
            EngOpts["platname"] = 'CUDA'
        else:
            EngOpts["platname"] = 'Reference'
        # Force crash if asking for the CUDA platform and force_cuda option is on
        # (because we don't want to inadvertently run using Reference platform)
        if EngOpts["platname"] == 'CUDA' and Fopts['force_cuda']:
            try: Platform.getPlatformByName('CUDA')
            except: raise RuntimeError('Forcing failure because CUDA platform unavailable')
        if threads > 1:
            logger.warn("Setting the number of threads will have no effect on OpenMM engine.\n")
    elif engname == "gromacs":
        # Gromacs-specific options
        EngOpts["gmxpath"] = Fopts["gmxpath"]
        EngOpts["gmxsuffix"] = Fopts["gmxsuffix"]
        EngOpts["gmx_top"] = Fopts["gmx_top"]
        EngOpts["gmx_mdp"] = Fopts["gmx_mdp"]
        if Fopts['force_cuda']: logger.warn("force_cuda option has no effect on Gromacs engine.")
        if Fopts['rpmd_beads'] > 0: raise RuntimeError("Gromacs cannot handle RPMD.")
        if Fopts['mts']: logger.warn("Gromacs not configured for multiple timestep integrator.")
        if Fopts['anisotropic']: logger.warn("Gromacs not configured for anisotropic box scaling.")
    elif engname == "tinker":
        EngOpts["tinkerpath"] = Fopts["tinkerpath"]
        EngOpts["tinker_key"] = Fopts["tinker_key"]

        # if Fopts['threads'] > 1: 
    printcool_dictionary(EngOpts, title="Engine options")

    # Number of threads, multiple timestep integrator, anisotropic box etc.
    # threads = Fopts.get('md_threads', 1)
    # mts = Fopts.get('mts_integrator', 0)
    # rpmd_beads = Fopts.get('rpmd_beads', 0)
    # force_cuda = Fopts.get('force_cuda', 0)
    # nbarostat = Fopts.get('n_mcbarostat', 25)
    # anisotropic = Fopts.get('anisotropic_box', 0)
    # minimize = Fopts.get('minimize_energy', 1)
    sys.exit()
    
    #----
    # Setting up MD simulations
    #----
    

    EngOpts["liquid"] = OrderedDict([("coords", liquid_fnm), ("mol", ML), ("pbc", True)])
    GenOpts = OrderedDict([('FF', FF)])
    if engname == "openmm":
        # OpenMM-specific options
        EngOpts["liquid"]["platname"] = 'CUDA'
        EngOpts["gas"]["platname"] = 'Reference'
        if force_cuda:
            try: Platform.getPlatformByName('CUDA')
            except: raise RuntimeError('Forcing failure because CUDA platform unavailable')
        if threads > 1: logger.warn("Setting the number of threads will have no effect on OpenMM engine.\n")
    elif engname == "gromacs":
        # Gromacs-specific options
        GenOpts["gmxpath"] = Fopts["gmxpath"]
        GenOpts["gmxsuffix"] = Fopts["gmxsuffix"]
        EngOpts["liquid"]["gmx_top"] = os.path.splitext(liquid_fnm)[0] + ".top"
        EngOpts["liquid"]["gmx_mdp"] = os.path.splitext(liquid_fnm)[0] + ".mdp"
        EngOpts["gas"]["gmx_top"] = os.path.splitext(gas_fnm)[0] + ".top"
        EngOpts["gas"]["gmx_mdp"] = os.path.splitext(gas_fnm)[0] + ".mdp"
        if force_cuda: logger.warn("force_cuda option has no effect on Gromacs engine.")
        if rpmd_beads > 0: raise RuntimeError("Gromacs cannot handle RPMD.")
        if mts: logger.warn("Gromacs not configured for multiple timestep integrator.")
        if anisotropic: logger.warn("Gromacs not configured for anisotropic box scaling.")
    elif engname == "tinker":
        # Tinker-specific options
        GenOpts["tinkerpath"] = Fopts["tinkerpath"]
        EngOpts["liquid"]["tinker_key"] = os.path.splitext(liquid_fnm)[0] + ".key"
        EngOpts["gas"]["tinker_key"] = os.path.splitext(gas_fnm)[0] + ".key"
        if force_cuda: logger.warn("force_cuda option has no effect on Tinker engine.")
        if rpmd_beads > 0: raise RuntimeError("TINKER cannot handle RPMD.")
        if mts: logger.warn("Tinker not configured for multiple timestep integrator.")
    EngOpts["liquid"].update(GenOpts)
    EngOpts["gas"].update(GenOpts)
    for i in EngOpts:
        printcool_dictionary(EngOpts[i], "Engine options for %s" % i)

    # Set up MD options
    # These are used in the function call to molecular_dynamics()

    MDOpts["liquid"] = OrderedDict([("nsteps", liquid_nsteps), ("timestep", liquid_timestep),
                                    ("temperature", temperature), ("pressure", pressure),
                                    ("nequil", liquid_nequil), ("minimize", minimize),
                                    ("nsave", int(1000 * liquid_intvl / liquid_timestep)),
                                    ("verbose", True), ('save_traj', Fopts['save_traj']), 
                                    ("threads", threads), ("anisotropic", anisotropic), ("nbarostat", nbarostat),
                                    ("mts", mts), ("rpmd_beads", rpmd_beads), ("faststep", faststep)])
    MDOpts["gas"] = OrderedDict([("nsteps", gas_nsteps), ("timestep", gas_timestep),
                                 ("temperature", temperature), ("nsave", int(1000 * gas_intvl / gas_timestep)),
                                 ("nequil", gas_nequil), ("minimize", minimize), ("threads", 1), ("mts", mts),
                                 ("rpmd_beads", rpmd_beads), ("faststep", faststep)])


    engines = []
    ## Setup and carry out simulations in chain
    for i in range(args.length):
        # Simulation files
        if engname == "gromacs":
            ndx_flag = False
            coords   = args.name + str(i+1) + ".gro"
            top_file = args.name + str(i+1) + ".top"
            mdp_file = args.name + str(i+1) + ".mdp"
            ndx_file = args.name + str(i+1) + ".ndx"
            if os.path.exists(ndx_file):
                ndx_flag = True
                
        mol = Molecule(coords)
        #----
        # Set coordinates and molecule for engine
        #----
        EngOpts = OrderedDict([("FF", FF),
                               ("pbc", True),
                               ("coords", coords),
                               ("mol", mol)])
    
        if engname == "gromacs":
            # Gromacs-specific options
            EngOpts["gmx_top"] = top_file
            EngOpts["gmx_mdp"] = mdp_file
            if ndx_flag:
                EngOpts["gmx_ndx"] = ndx_file
                
        printcool_dictionary(EngOpts)
                                
        # Create engine objects and store them for subsequent analysis.
        s = Engine(name=args.name+str(i+1), **EngOpts)
                
        #=====================#
        # Run the simulation. #
        #=====================#
        MDOpts = OrderedDict([("nsteps", args.nsteps),
                              ("nequil", args.nequil)])

        printcool("Molecular dynamics simulation", color=4, bold=True)
        s.md(verbose=True, **MDOpts)
                                    
        engines.append(s)
    
    #======================================================================#
    # Extract the quantities of interest from the MD simulations and dump  #
    # the results to file.                                                 #
    # =====================================================================#    
    results = OrderedDict()        
    for q in args.quantities:
        logger.info("Extracting %s...\n" % q)

        # Initialize quantity
        objstr = "Quantity_" + q.capitalize()
        dm     = il.import_module('..quantity',
                                  package='forcebalance.quantity')
            
        Quantity = getattr(dm, objstr)(engname, args.temperature, args.pressure)
            
        Q, Qerr, Qgrad = Quantity.extract(engines, FF, mvals, h, pgrad, AGrad)
                    
        results.setdefault("values", []).append(Q)
        results.setdefault("errors", []).append(Qerr)
        results.setdefault("grads",  []).append(Qgrad)
            
        logger.info("Finished!\n")
            
        # Print out results for the quantity and its derivative.
        Sep = printcool(("%s: % .4f +- % .4f \nAnalytic Derivative:"
                              % (q.capitalize(), Q, Qerr)))
        FF.print_map(vals=Qgrad)
            
    # Dump results to file
    logger.info("Writing final force field.\n")
    pvals = FF.make(mvals)
    
    logger.info("Writing all simulation data to disk.\n")
    with wopen('md_result.p') as f:
        lp_dump((np.asarray(results["values"]),
                 np.asarray(results["errors"]),
                 np.asarray(results["grads"])), f)
    
if __name__ == "__main__":
    main()

