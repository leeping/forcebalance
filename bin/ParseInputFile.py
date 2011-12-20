#!/usr/bin/env python

""" @package ParseInputFile Read in ForceBalance input file and print it back out."""

from nifty import printcool_dictionary
from parser import parse_imports
import sys

def main():
    """Input file parser for ForceBalance program.  We will simply read the options and print them back out.
    
    """
    print "\x1b[1;98mCalling Input File Parser as a standalone script\x1b[0m\n"
    if len(sys.argv) != 2:
        print "Please call this script with one argument - that is the input file"
        sys.exit(1)
    else:
        options, sim_opts = parse_inputs(sys.argv[1])
        printcool_dictionary(options,"General options")
        for this_sim_opt in sim_opts:
            printcool_dictionary(this_sim_opt,"Simulation options for %s" % this_sim_opt['name'])

if __name__ == "__main__":
    main()
