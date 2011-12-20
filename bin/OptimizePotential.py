#!/usr/bin/env python

""" @package OptimizePotential

Executable script for starting ForceBalance. """

import sys
from project import Project

def main():
    """ Instantiate a ForceBalance project and call the optimizer. """
    print "\x1b[1;97m Welcome to ForceBalance version 0.11.0! =D\x1b[0m"
    if len(sys.argv) != 2:
        print "Please call this program with only one argument - the name of the input file."
        sys.exit(1)
    P = Project(sys.argv[1])
    P.Run()

if __name__ == "__main__":
    main()
