""" @package vibration Vibrational mode fitting module

@author Lee-Ping Wang
@date 08/2012
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
from _assign import Assign
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
        self.set_option(sim_opts,'wavenumber_tol','denom')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The vdata.txt file that contains the vibrations.
        self.vfnm = os.path.join(self.simdir,"vdata.txt")
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)

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
                xyz = zeros((self.na, 3), dtype=float)
                cn = ln + 1
            elif ln == cn:
                pass
            elif an < self.na and len(s) == 4:
                xyz[an, :] = array([float(i) for i in s[1:]])
                an += 1
            elif len(s) == 1:
                self.ref_eigvals.append(float(s[0]))
                self.ref_eigvecs.append(zeros((self.na, 3), dtype=float))
                an = 0
            elif len(s) == 3:
                self.ref_eigvecs[-1][an, :] = array([float(i) for i in s])
                an += 1
            elif len(s) == 0:
                pass
            else:
                print line
                raise Exception("This line doesn't comply with our vibration file format!")
            ln += 1
        self.ref_eigvals = array(self.ref_eigvals)
        self.ref_eigvecs = array(self.ref_eigvecs)

        return

    def prepare_temp_directory(self, options, sim_opts):
        """ Prepare the temporary directory, by default does nothing (gmxx2 needs it) """
        return
        
    def indicate(self):
        """ Print qualitative indicator. """
        print "\rSim: %-15s" % self.name, 
        print "Frequencies (wavenumbers), Reference:", self.ref_eigvals,
        print "Calculated:", self.calc_eigvals,
        print "Objective = %.5e" % self.objective
        return

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':zeros(self.FF.np, dtype=float), 'H':zeros((self.FF.np, self.FF.np), dtype=float)}
        def get_eigvals(mvals_):
            self.FF.make(mvals_)
            eigvals, eigvecs = self.vibration_driver()
            # Put reference eigenvectors in the rows and calculated eigenvectors in the columns.
            # Square the dot product (pointing in opposite direction is still the same eigenvector)
            # Convert to integer for the "Assign" subroutine, subtract from a million.
            a = array([[int(1e6*(1.0-min(1.0,dot(v1.flatten(),v2.flatten())**2))) for v2 in self.ref_eigvecs] for v1 in eigvecs])
            # In the matrix that we constructed, these are the column numbers (reference mode numbers) 
            # that are mapped to the row numbers (calculated mode numbers)
            c2r = Assign(a)
            return eigvals[c2r]

        calc_eigvals = get_eigvals(mvals)
        D = calc_eigvals - self.ref_eigvals
        dV = zeros((self.FF.np,len(calc_eigvals)),dtype=float)

        if AGrad or AHess:
            for p in range(self.FF.np):
                dV[p,:], _ = f12d3p(fdwrap(get_eigvals, mvals, p), h = self.h, f0 = calc_eigvals)
                
        Answer['X'] = dot(D,D) / self.denom**2
        for p in range(self.FF.np):
            Answer['G'][p] = 2*dot(D, dV[p,:]) / self.denom**2
            for q in range(self.FF.np):
                Answer['H'][p,q] = 2*dot(dV[p,:], dV[q,:]) / self.denom**2

        if not in_fd():
            self.calc_eigvals = calc_eigvals
            self.objective = Answer['X']

        return Answer
