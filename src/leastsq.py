""" @package forcebalance.abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt, abs, max
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from forcebalance.optimizer import Counter

CHECK_BASIS = False
def CheckBasis():
    global CHECK_BASIS
    return CHECK_BASIS

LAST_MVALS = None
def LastMvals():
    global LAST_MVALS
    return LAST_MVALS

class LeastSquares(Target):

    """ Subclass of Target for general least squares fitting. """
    
    def __init__(self,options,tgt_opts,forcefield):
        super(LeastSquares,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.set_option(tgt_opts,'shots','ns')
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Which parameters are differentiated?
        self.call_derivatives = [True for i in range(forcefield.np)]

    def indicate(self):
        #RMSD = sqrt(mean(self.D ** 2))
        MAD = mean(abs(self.D))
        print "\rTarget: %-15s" % self.name, 
        print "MeanAbsErr/MeanExact: %.5e Objective = %.5e" % (MAD / self.MAQ, self.objective)
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
        global LAST_MVALS, CHECK_BASIS
        # print mvals
        # print LAST_MVALS
        # print mvals == LAST_MVALS
        if LAST_MVALS == None or not (mvals == LAST_MVALS).all():
            CHECK_BASIS = False
        else:
            CHECK_BASIS = False
        Answer = {}
        Fac = 1000000
        ## Dictionary for derivative terms
        dM = {}
        # Create the new force field!!
        np = len(mvals)
        G = zeros(np,dtype=float)
        H = zeros((np,np),dtype=float)
        pvals = self.FF.make(mvals)
        if float('Inf') in pvals:
            return {'X' : 1e10, 'G' : G, 'H' : H}
        Ans = self.driver()
        W = Ans[:,2]
        M = Ans[:,1]
        Q = Ans[:,0]
        D = M - Q

        self.MAQ = mean(abs(Q))

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
            self.D = D
            self.objective = Answer['X']
            LAST_MVALS = mvals.copy()
        return Answer

