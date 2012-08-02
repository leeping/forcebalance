""" @package vibration Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, warn_press_key
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt, hstack
from fitsim import FittingSimulation
from molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from finite_difference import fdwrap, f1d2p, f12d3p, in_fd
#from _increment import Vibration_Build

class Vibration(FittingSimulation):

    """ Subclass of FittingSimulation for fitting force fields to vibrational spectra (from experiment or theory).

    Currently Tinker is supported.

    """
    
    def __init__(self,options,sim_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Vibration,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.simdir,"qdata.txt")
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)

    def read_reference_data(self):
        """ Read the reference ab initio data from a file. """
        return

    def prepare_temp_directory(self, options, sim_opts):
        """ Prepare the temporary directory, by default does nothing (gmxx2 needs it) """
        return
        
    def indicate(self):
        """ Print qualitative indicator. """
        return

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':zeros(self.FF.np, dtype=float), 'H':zeros((self.FF.np, self.FF.np), dtype=float)}
        self.FF.make(mvals)
        self.vibration_driver()
        return Answer
