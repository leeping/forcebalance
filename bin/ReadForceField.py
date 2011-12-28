#!/usr/bin/env python

"@package ReadForceField Read force field from a file and print information out."

from forcebalance.forcefield import FF
from forcebalance.nifty import printcool
from sys import argv
import os

def main():
    ## Set some basic options.  Note that 'forcefield' requires 'ffdir'
    ## which indicates the relative path of the force field.
    options = {'forcefield':argv[1:],
               'ffdir':''}
    MyFF = FF(options)
    print MyFF.tm

if __name__ == "__main__":
    main()
