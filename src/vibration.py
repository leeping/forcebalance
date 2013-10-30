""" @package forcebalance.vibration Vibrational mode fitting module

@author Lee-Ping Wang
@date 08/2012
"""

import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, warn_press_key
import numpy as np
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from _assign import Assign
from collections import OrderedDict
#from _increment import Vibration_Build

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Vibration(Target):

    """ Subclass of Target for fitting force fields to vibrational spectra (from experiment or theory).

    Currently Tinker is supported.

    """
    
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Vibration,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts,'wavenumber_tol','denom')
        self.set_option(tgt_opts,'permute','permute')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The vdata.txt file that contains the vibrations.
        self.vfnm = os.path.join(self.tgtdir,"vdata.txt")
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,tgt_opts)

        if self.FF.rigid_water:
            raise Exception('This class cannot be used with rigid water molecules.')

    def read_reference_data(self):
        """ Read the reference vibrational data from a file. """
        ## Number of atoms
        self.na = -1
        self.ref_eigvals = []
        self.ref_eigvecs = []
        an = 0
        ln = 0
        cn = -1
        for line in open(self.vfnm):
            line = line.split('#')[0] # Strip off comments
            s = line.split()
            if len(s) == 1 and self.na == -1:
                self.na = int(s[0])
                xyz = np.zeros((self.na, 3))
                cn = ln + 1
            elif ln == cn:
                pass
            elif an < self.na and len(s) == 4:
                xyz[an, :] = np.array([float(i) for i in s[1:]])
                an += 1
            elif len(s) == 1:
                self.ref_eigvals.append(float(s[0]))
                self.ref_eigvecs.append(np.zeros((self.na, 3)))
                an = 0
            elif len(s) == 3:
                self.ref_eigvecs[-1][an, :] = np.array([float(i) for i in s])
                an += 1
            elif len(s) == 0:
                pass
            else:
                logger.info(line + '\n')
                raise Exception("This line doesn't comply with our vibration file format!")
            ln += 1
        self.ref_eigvals = np.array(self.ref_eigvals)
        self.ref_eigvecs = np.array(self.ref_eigvecs)

        return

    def prepare_temp_directory(self, options, tgt_opts):
        """ Prepare the temporary directory, by default does nothing """
        return
        
    def indicate(self):
        """ Print qualitative indicator. """
        banner = "Frequencies (wavenumbers)"
        headings = ["Mode #", "Reference", "Calculated"]
        data = OrderedDict([(i, [self.ref_eigvals[i], self.calc_eigvals[i]]) for i in range(len(self.ref_eigvals))])
        self.printcool_table(data, headings, banner)
        return

    def vibration_driver(self):
        if hasattr(self, 'engine') and hasattr(self.engine, 'normal_modes'):
            return self.engine.normal_modes()
        else:
            raise NotImplementedError('Normal mode calculation not supported, try using a different engine')

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        def get_eigvals(mvals_):
            self.FF.make(mvals_)
            eigvals, eigvecs = self.vibration_driver()
            # Put reference eigenvectors in the rows and calculated eigenvectors in the columns.
            # Square the dot product (pointing in opposite direction is still the same eigenvector)
            # Convert to integer for the "Assign" subroutine, subtract from a million.
            if self.permute:
                a = np.array([[int(1e6*(1.0-min(1.0,np.dot(v1.flatten(),v2.flatten())**2))) for v2 in self.ref_eigvecs] for v1 in eigvecs])
                # In the matrix that we constructed, these are the column numbers (reference mode numbers) 
                # that are mapped to the row numbers (calculated mode numbers)
                c2r = Assign(a)
                return eigvals[c2r]
            else:
                return eigvals

        calc_eigvals = get_eigvals(mvals)
        D = calc_eigvals - self.ref_eigvals
        dV = np.zeros((self.FF.np,len(calc_eigvals)))

        if AGrad or AHess:
            for p in range(self.FF.np):
                dV[p,:], _ = f12d3p(fdwrap(get_eigvals, mvals, p), h = self.h, f0 = calc_eigvals)
                
        Answer['X'] = np.dot(D,D) / self.denom**2
        for p in range(self.FF.np):
            Answer['G'][p] = 2*np.dot(D, dV[p,:]) / self.denom**2
            for q in range(self.FF.np):
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:]) / self.denom**2

        if not in_fd():
            self.calc_eigvals = calc_eigvals
            self.objective = Answer['X']

        return Answer
