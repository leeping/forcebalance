""" @package abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt, abs, max
from fitsim import FittingSimulation
from molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from optimizer import Counter

class LeastSquares(FittingSimulation):

    """ Subclass of FittingSimulation for general least squares fitting. """
    
    def __init__(self,options,sim_opts,forcefield):
        super(LeastSquares,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.set_option(sim_opts,'shots','ns')
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)
        ## Which parameters are differentiated?
        self.call_derivatives = [True for i in range(forcefield.np)]

    def prepare_temp_directory(self, options, sim_opts):
        """ Prepare the temporary directory, by default does nothing (gmxx2 needs it) """
        return
        
    def indicate(self):
        print "\rSim: %-15s" % self.name, 
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
        ## Dictionary for derivative terms
        dM = {}
        # Create the new force field!!
        np = len(mvals)
        G = zeros(np,dtype=float)
        H = zeros((np,np),dtype=float)
        pvals = self.FF.make(mvals,self.usepvals)
        if float('Inf') in pvals:
            return {'X' : 1e10, 'G' : G, 'H' : H}
        Ans = self.driver()
        W = Ans[:,2]
        M = Ans[:,1]
        Q = Ans[:,0]
        D = M - Q
        ns = len(M)
        # Wrapper to the driver, which returns just the part that changes.
        def callM(mvals_):
            self.FF.make(mvals_)
            Ans2 = self.driver()
            M_ = Ans2[:,1]
            D_ = M_ - Q
	    return Ans2[:,1]
        if AGrad:
            # Leaving comment here if we want to reintroduce second deriv someday.
            #     dM[p,:], ddM[p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
            for p in range(np):
                if self.call_derivatives[p] == False: continue
                dM_arr = f1d2p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
                if max(abs(dM_arr)) == 0.0 and Counter() == 0:
                    print "\r Simulation %s will skip over parameter %i in subsequent steps" % (self.name, p)
                    self.call_derivatives[p] = False
                else:
                    dM[p] = dM_arr.copy()
	Objective = dot(W, D**2) * Fac
        if AGrad:
            for p in range(np):
                if self.call_derivatives[p] == False: continue
                G[p] = 2 * dot(W, D*dM[p])
                if not AHess: continue
                H[p, p] = 2 * dot(W, dM[p]**2)
                for q in range(p):
                    if self.call_derivatives[q] == False: continue
                    GNP = 2 * dot(W, dM[p] * dM[q])
                    H[q,p] = GNP
                    H[p,q] = GNP
        G *= Fac
        H *= Fac
        Answer = {'X':Objective, 'G':G, 'H':H}
        if not in_fd():
            self.objective = Answer['X']
        return Answer

