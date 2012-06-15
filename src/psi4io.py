""" @package psi4io PSI4 force field input/output.

This serves as a good template for writing future force matching I/O
modules for other programs because it's so simple.

@author Lee-Ping Wang
@date 01/2012
"""

import os
from re import match, sub, split, findall
from nifty import isint, isfloat, _exec
import numpy as np
from leastsq import LeastSquares

class LRDF_Psi4(LeastSquares):

    def __init__(self,options,sim_opts,forcefield):
        super(LRDF_Psi4,self).__init__(options,sim_opts,forcefield)

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        os.symlink(os.path.join(self.root,self.simdir,"input.dat"),os.path.join(abstempdir,"input.dat"))

    def driver(self):
        ## Actually run PSI4.
        _exec("psi4", print_command=False)
        return np.array([[float(i) for i in line.split()] for line in open("objective.dat").readlines()])
