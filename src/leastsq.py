""" @package abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt
from fitsim import FittingSimulation
from molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from finite_difference import fdwrap, f1d2p, f12d3p, in_fd

class LeastSquares(FittingSimulation):

    """ Subclass of FittingSimulation for general least squares fitting. """
    
    def __init__(self,options,sim_opts,forcefield):
        super(LeastSquares,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.ns            = sim_opts['shots']
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)

    def prepare_temp_directory(self, options, sim_opts):
        """ Prepare the temporary directory, by default does nothing (gmxx2 needs it) """
        return
        
    def indicate(self):
        print "Sim: %-15s" % self.name, 
        print "Objective = %.5e" % self.objective
        return

    def get(self, mvals, AGrad=False, AHess=False):
        """
        LPW 05-30-2012
        
        This subroutine builds the objective function (and optionally
        its derivatives) from a general software.  

        This subroutine interfaces with simulation software 'drivers'.
        The driver is expected to give exact values, fitting values, and weights.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        """
        Answer = {}
        Fac = 1000000
        # Create the new force field!!
        np = len(mvals)
        pvals = self.FF.make(mvals,self.usepvals)
        Ans = self.driver()
        W = Ans[:,2]
        M = Ans[:,1]
        Q = Ans[:,0]
        D = M - Q
        ns = len(M)
        # Wrapper to the driver, which returns just the part that changes.
        def callM(mvals_):
            self.FF.make(mvals_)
            return self.driver()[:,1]
        if AGrad:
            dM = zeros((np,ns),dtype=float)
            ddM = zeros((np,ns),dtype=float)
            # for p in range(np):
            #     dM[p,:], ddM[p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
            for p in range(np):
                dM[p,:] = f1d2p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
        Objective = dot(W, D**2) * Fac
        G = zeros(np,dtype=float)
        H = zeros((np,np),dtype=float)
        if AGrad:
            for p in range(np):
                G[p] = 2 * dot(W, D*dM[p,:])
                if not AHess: continue
                H[p, p] = 2 * dot(W, D*ddM[p,:] + dM[p,:]**2)
                for q in range(p):
                    GNP = 2 * dot(W, dM[p,:] * dM[q,:])
                    H[q,p] = GNP
                    H[p,q] = GNP
        G *= Fac
        H *= Fac
        Answer = {'X':Objective, 'G':G, 'H':H}
        if not in_fd():
            self.objective = Answer['X']
        return Answer

