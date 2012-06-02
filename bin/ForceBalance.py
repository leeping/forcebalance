#!/usr/bin/env python

""" @package ForceBalance

Executable script for starting ForceBalance. """

import sys
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer

def Run_ForceBalance(input_file):
    """ Create instances of ForceBalance components and run the optimizer.

    The triumvirate, trifecta, or trinity of components are:
    - The force field
    - The objective function
    - The optimizer
    Cipher: "All I gotta do here is pull this plug... and there you have to watch Apoc die"
    Apoc: "TRINITY" *chunk*

    The force field is a class defined in forcefield.py.
    The objective function is a combination of fitting simulation classes and a penalty function class.
    The optimizer is a class defined in this file.
    """
    ## The general options and simulation options that come from parsing the input file
    options, sim_opts = parse_inputs(input_file)
    ## The force field component of the project
    forcefield  = FF(options)
    ## The objective function
    objective   = Objective(options, sim_opts, forcefield)
    ## The optimizer component of the project
    optimizer   = Optimizer(options, objective, forcefield)
    ## Actually run the optimizer.
    optimizer.Run()

def main():
    print "\x1b[1;97m Welcome to ForceBalance version 0.12! =D\x1b[0m"
    if len(sys.argv) != 2:
        print "Please call this program with only one argument - the name of the input file."
        sys.exit(1)
    Run_ForceBalance(sys.argv[1])

if __name__ == "__main__":
    main()
