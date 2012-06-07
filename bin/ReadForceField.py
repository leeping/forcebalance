#!/usr/bin/env python

"@package ReadForceField Read force field from a file and print information out."

from forcebalance.forcefield import FF
from forcebalance.nifty import printcool
from sys import argv
import os
import numpy as np

def main():
    ## Set some basic options.  Note that 'forcefield' requires 'ffdir'
    ## which indicates the relative path of the force field.
    options = {'forcefield':[argv[1]],
               'ffdir':'',
               'priors':{},
               'constrain_charge':True}
    MyFF = FF(options)
    
    Prec=int(argv[2])
    MyFF.make(np.zeros(len(MyFF.pvals0)),False,'NewFF',precision=Prec)

if __name__ == "__main__":
    main()
