#!/usr/bin/env python

"""
md_chain
========

This script is a part of ForceBalance and runs a chain of simulations that can
be used to compute any thermodynamic properties.

This script is meant to be launched automatically by ForceBalance.

"""

#==================#
#| Global Imports |#
#==================#

import os
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

parser = argparse.ArgumentParser()
parser.add_argument('quantities', nargs='+',
                    help='Quantities to calculate')
parser.add_argument('--engine', default='gromacs',
                    help='MD program that we are using; choose "gromacs"')
parser.add_argument('--length', type=int, default=1,
                    help='Length of simulation chain')
parser.add_argument('--name', default='sim',
                    help='Default simulation names')
parser.add_argument('--temperature', type=float, default=None,
                    help='Simulation temperature')
parser.add_argument('--pressure', type=float, default=None,
                    help='Simulation pressure')
parser.add_argument('--nequil', type=int, default=0,
                    help='Number of steps for equilibration run')
parser.add_argument('--nsteps', type=int, default=0,
                    help='Number of steps for production run')
args = parser.parse_args()

engname = args.engine.lower() # Name of the engine
if engname == "gromacs":
    from forcebalance.gmxio import GMX
    Engine = GMX
else:
    raise Exception('Only GROMACS supported at this time.')

def main():
    
    """Usage:
    
    (gmxprefix.sh) md_chain.py <list of quantities>
                               --engine <gromacs>
                               --length <n>
                               --name <name>
                               --temperature <T>
                               --pressure <p>
                               --nequil <nequil>
                               --nsteps <nsteps>
        
    This program is meant to be called automatically by ForceBalance.
    
    """
    printcool("ForceBalance simulation using engine: %s" % engname.upper(),
              color=4, bold=True)
    #----
    # Load the ForceBalance pickle file which contains:
    #----
    # - Force field object
    # - Optimization parameters
    # - Options from the Target object that launched this simulation
    # - Switch for whether to evaluate analytic derivatives.
    FF, mvals, TgtOptions, AGrad = lp_load('forcebalance.p')
    FF.ffdir = '.'
    # Write the force field file.
    FF.make(mvals)

    #----
    # Load the options that are set in the ForceBalance input file.
    #----
    # Finite difference step size
    h = TgtOptions['h']
    pgrad = TgtOptions['pgrad']
    
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
    lp_dump((np.asarray(results["values"]),
             np.asarray(results["errors"]),
             np.asarray(results["grads"])), 'md_result.p')
    
if __name__ == "__main__":
    main()

