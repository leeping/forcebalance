""" @package forcebalance.abinitio_pairwise Ab-initio fitting module with pairwise energies.

@author Chapin Cavender
@date 05/2024
"""
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
import os
import shutil
from forcebalance.abinitio import AbInitio, plot_qm_vs_mm
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohr2ang, warn_press_key, warn_once, pvec1d, commadash, uncommadash, isint
import numpy as np
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import defaultdict, OrderedDict
import itertools

from forcebalance.output import getLogger
logger = getLogger(__name__)

class AbInitioPairwise(AbInitio):

    """ Subclass of AbInitio with objective function composed of pairwise energy differences."""

    def __init__(self,options,tgt_opts,forcefield):
        """
        Initialization; set up pairwise references energies and weights.
        """

        # Initialize the base class
        super(AbInitioPairwise,self).__init__(options,tgt_opts,forcefield)

        # Raise an error if the target tries to fit forces, net forces, or net
        # torques
        if self.force or self.use_nft:
            raise(
                RuntimeError(
                    "Fitting using forces, net forces, or net torques is not "
                    "implemented for this target."
                )
            )

        # Set up pairwise reference energies and weights
        N_pairs = int(self.ns * (self.ns - 1) / 2)
        self.eqm_pairs = np.zeros(N_pairs)
        self.boltz_wt_pairs = np.ones(N_pairs)

        for i, pair in enumerate(itertools.combinations(range(self.ns), 2)):
            self.eqm_pairs[i] = self.eqm[pair[0]] - self.eqm[pair[1]]
            self.boltz_wt_pairs[i] = np.sqrt(
                self.boltz_wts[pair[0]] * self.boltz_wts[pair[1]]
            )

        self.boltz_wt_pairs /= self.boltz_wt_pairs.sum()


    def get_energy_force(self, mvals, AGrad=False, AHess=False):
        """
        This code computes the least squares objective function for pairwise
        energy differences. The numerator is a weighted sum of squared
        differences between (E_MM,i - E_MM,j) and (E_QM,i - E_QM,j) for all
        pairs of grid points (i, j). The weighted variance of the QM energies
        <E_QM^2>-<E_QM>^2 is the denominator. The indicators are set to the
        square roots of the numerator and denominator above.

        Fitting using forces, net forces, or net torques is not implemented for
        this target.

        In equation form, the objective function is given by:

        \[ = {\frac{{\left( {\sum\limits_{i,j} {{w_i}{{\left(
        {\left( E_i^{MM} - E_j^{MM} \right) - \left( E_i^{QM} - E_j^{QM} \right)}
        \right)}^2}} } \right)}} {{\sum\limits_{i,j} {{w_i}{{\left(
        {E_i^{QM} - {{\bar E}^{QM}}} \right)}^2}} }}} \]

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        """
        Answer = {}
        # Create the new force field!!
        pvals = self.FF.make(mvals)

        # Number of atoms containing forces being fitted
        nat  = len(self.fitatoms)
        # Number of net forces on molecules
        nnf  = self.nnf
        # Number of torques on molecules
        ntq  = self.ntq
        # Basically the size of the matrix
        NCP1 = 3*nat+1
        NParts = 1
        NP   = self.FF.np
        NS   = self.ns
        #==============================================================#
        #            STEP 1: Form all of the arrays.                   #
        #==============================================================#
        if (self.w_energy == 0.0 and self.w_force == 0.0 and self.w_netforce == 0.0 and self.w_torque == 0.0):
            AGrad = False
            AHess = False
        # Sum of all the weights
        Z       = 0.0
        # All vectors with NCP1 elements are ordered as
        # [E F_1x F_1y F_1z F_2x ... NF_1x NF_1y ... TQ_1x TQ_1y ... ]
        # Vector of QM-quantities
        Q = np.zeros(NCP1)
        # Mean quantities over the trajectory
        M0    = np.zeros(NCP1)
        Q0    = np.zeros(NCP1)
        X0    = np.zeros(NCP1)
        # The mean squared QM-quantities
        QQ0    = np.zeros(NCP1)
        # Derivatives of the MM-quantity
        M_p     = np.zeros((NP,NCP1))
        M_pp    = np.zeros((NP,NCP1))
        # Means of gradients
        M0_p  = np.zeros((NP,NCP1))
        M0_pp = np.zeros((NP,NCP1))
        # Vector of objective function terms
        SPX = np.zeros(NCP1)
        if AGrad:
            SPX_p = np.zeros((NP,NCP1))
            # Derivatives of "parts" of objective functions - i.e.
            # the sum is taken over the components of force, net force, torque
            # but the components haven't been weighted and summed.
            X2_Parts_p = np.zeros((NP,NParts))
        if AHess:
            SPX_pq = np.zeros((NP,NP,NCP1))
            X2_Parts_pq = np.zeros((NP,NP,NParts))
        # Storage of the MM-quantities and derivatives for all snapshots.
        # This saves time because we don't need to execute the external program
        # once per snapshot, but requires memory.
        M_all = np.zeros((NS,NCP1))
        if AGrad and self.all_at_once:
            dM_all = np.zeros((NS,NP,NCP1))
            ddM_all = np.zeros((NS,NP,NCP1))
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        if self.all_at_once:
            logger.debug("\rExecuting\r")
            M_all = self.energy_force_transform()
            if self.energy_mode == 'qm_minimum':
                M_all[:, 0] -= M_all[self.smin, 0]
            if AGrad or AHess:
                def callM(mvals_):
                    logger.debug("\r")
                    pvals = self.FF.make(mvals_)
                    return self.energy_force_transform()
                for p in self.pgrad:
                    dM_all[:,p,:], ddM_all[:,p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M_all)
                    if self.energy_mode == 'qm_minimum':
                        dM_all[:, p, 0] -= dM_all[self.smin, p, 0]
                        ddM_all[:, p, 0] -= ddM_all[self.smin, p, 0]
        else:
            for i in range(NS):
                if i % 100 == 0:
                    logger.debug("Shot %i\r" % i)
                M = self.energy_force_transform_one(i)
                M_all[i,:] = M.copy()

                if not AGrad: continue
                for p in self.pgrad:
                    def callM(mvals_):
                        if i % 100 == 0:
                            logger.debug("\r")
                        pvals = self.FF.make(mvals_)
                        return self.energy_force_transform_one(i)
                    dM_all[i, p, :], ddM_all[i, p, :] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
        for i, pair in enumerate(itertools.combinations(range(NS), 2)):
            if i % 100 == 0:
                logger.debug("\rIncrementing quantities for pair %i, %i\r" % pair)
            # Build Boltzmann weights and increment partition function.
            P   = self.boltz_wt_pairs[i]
            Z  += P
            # Load reference (QM) data
            Q[0] = self.eqm_pairs[i]
            QQ     = Q*Q
            # Load MM quantities from M_all array
            M = M_all[pair[0]] - M_all[pair[1]]
            # MM pair - QM pair difference
            X     = M-Q
            # For asymmetric fit, MM pair energy differences with opposite sign
            # as the corresponding QM pair energy difference are given a boost
            # factor.
            boost = self.energy_asymmetry if ((M[0] < 0.0) != (Q[0] < 0.0)) else 1.0
            # Increment the average quantities
            # The [0] indicates that we are fitting the RMS force and not the RMSD
            # (without the covariance, subtracting a mean force doesn't make sense.)
            # The rest of the array is empty.
            M0[0] += P*M[0]
            Q0[0] += P*Q[0]
            X0[0] += P*X[0]
            # We store all elements of the mean-squared QM quantities.
            QQ0 += P*QQ
            # Increment the objective function.
            Xi     = X**2
            Xi[0] *= boost
            # SPX contains the sum over snapshots
            SPX += P * Xi
            #==============================================================#
            #      STEP 2a: Increment gradients and mean quantities.       #
            #==============================================================#
            for p in self.pgrad:
                if not AGrad: continue
                M_p[p] = dM_all[pair[0], p] - dM_all[pair[1], p]
                M_pp[p] = ddM_all[pair[0], p] - ddM_all[pair[1], p]
                if all(M_p[p] == 0): continue
                M0_p[p][0]  += P * M_p[p][0]
                Xi_p        = 2 * X * M_p[p]
                Xi_p[0]    *= boost
                SPX_p[p] += P * Xi_p
                if not AHess: continue
                # This formula is more correct, but perhapsively convergence is slower.
                #Xi_pq       = 2 * (M_p[p] * M_p[p] + X * M_pp[p])
                # Gauss-Newton formula for approximate Hessian
                Xi_pq       = 2 * (M_p[p] * M_p[p])
                Xi_pq[0]   *= boost
                SPX_pq[p,p] += P * Xi_pq
                for q in range(p):
                    if all(M_p[q] == 0): continue
                    if q not in self.pgrad: continue
                    Xi_pq          = 2 * M_p[p] * M_p[q]
                    Xi_pq[0]      *= boost
                    SPX_pq[p,q] += P * Xi_pq

        #==============================================================#
        #         STEP 2b: Write energies and forces to disk.          #
        #==============================================================#
        M_all_print = M_all.copy()
        Q_all_print = col(self.eqm)
        if self.energy_mode == 'average':
            M_all_print[:,0] -= np.average(M_all_print[:,0], weights=self.boltz_wts)
            Q_all_print[:,0] -= np.average(Q_all_print[:,0], weights=self.boltz_wts)
        elif self.energy_mode == 'qm_minimum':
            M_all_print[:,0] -= M_all_print[self.smin,0]
            Q_all_print[:,0] -= Q_all_print[self.smin,0]
        if self.writelevel > 1:
            np.savetxt('M.txt',M_all_print)
            np.savetxt('Q.txt',Q_all_print)
        if self.writelevel > 0:
            EnergyComparison = np.hstack((col(np.arange(NS)),
                                          col(Q_all_print[:,0]),
                                          col(M_all_print[:,0]),
                                          col(M_all_print[:,0])-col(Q_all_print[:,0]),
                                          col(self.boltz_wts)))
            np.savetxt("EnergyCompare.txt", EnergyComparison, fmt=" %12i  % 12.5f  % 12.5f  % 13.5f  % 12.5e", header="%11s  %12s  %12s  %13s  %12s" % ("Num", "QMEnergy", "MMEnergy", "DeltaE(MM-QM)", "Weight"))
            if self.writelevel > 1:
                plot_qm_vs_mm(Q_all_print[:,0], M_all_print[:,0],
                              M_orig=self.M_orig[:,0] if self.M_orig is not None else None,
                              title='Abinitio '+self.name)
            if self.M_orig is None:
                self.M_orig = M_all_print.copy()

        #==============================================================#
        #            STEP 3: Build the objective function.             #
        #==============================================================#

        logger.debug("Done with snapshots, building objective function now\r")
        W_Components = np.zeros(NParts)
        W_Components[0] = self.w_energy
        if np.sum(W_Components) > 0 and self.w_normalize:
            W_Components /= np.sum(W_Components)

        if self.energy_rms_override != 0.0:
            QQ0[0] = self.energy_rms_override ** 2
            Q0[0] = 0.0

        def compute_objective(SPX_like,divide=1,L=None,R=None,L2=None,R2=None):
            a = 0
            n = 1
            X2E = compute_objective_part(SPX_like,QQ0,Q0,Z,a,n,energy=False,subtract_mean=(self.energy_mode=='average'),
                                         divide=divide,L=L,R=R,L2=L2,R2=R2)
            objs = [X2E]
            return np.array(objs)

        # The objective function components (i.e. energy, force, net force, torque)
        X2_Components = compute_objective(SPX,L=X0,R=X0)
        # The squared residuals before they are normalized
        X2_Physical = compute_objective(SPX,divide=0,L=X0,R=X0)
        # The normalization factors
        X2_Normalize = compute_objective(SPX,divide=2,L=X0,R=X0)
        # The derivatives of the objective function components
        for p in self.pgrad:
            if not AGrad: continue
            X2_Parts_p[p,:] = compute_objective(SPX_p[p],L=2*X0,R=M0_p[p])
            if not AHess: continue
            X2_Parts_pq[p,p,:] = compute_objective(SPX_pq[p,p],L=2*M0_p[p],R=M0_p[p],L2=2*X0,R2=M0_pp[p])
            for q in range(p):
                if q not in self.pgrad: continue
                X2_Parts_pq[p,q,:] = compute_objective(SPX_pq[p,q],L=2*M0_p[p],R=M0_p[q])
                # Get the other half of the Hessian matrix.
                X2_Parts_pq[q,p,:] = X2_Parts_pq[p,q,:]
        # The objective function as a weighted sum of the components
        X2   = np.dot(X2_Components, W_Components)
        # Derivatives of the objective function
        G = np.zeros(NP)
        H = np.zeros((NP,NP))
        for p in self.pgrad:
            if not AGrad: continue
            G[p] = np.dot(X2_Parts_p[p], W_Components)
            if not AHess: continue
            for q in self.pgrad:
                H[p,q] = np.dot(X2_Parts_pq[p,q], W_Components)

        #==============================================================#
        #                STEP 3a: Build the indicators.                #
        #==============================================================#
        # Save values to qualitative indicator if not inside finite difference code.
        if not in_fd():
            # Contribution from energy and force parts.
            tw = self.w_energy + self.w_force + self.w_netforce + self.w_torque + self.w_resp
            self.e_trm = X2_Components[0]
            self.e_ctr = X2_Components[0]*W_Components[0]
            if self.w_normalize: self.e_ctr /= tw
            self.e_ref = np.sqrt(X2_Normalize[0])
            self.e_err = np.sqrt(X2_Physical[0])
            self.e_err_pct = self.e_err/self.e_ref

            pvals = self.FF.make(mvals) # Write a force field that isn't perturbed by finite differences.
        Answer = {'X':X2, 'G':G, 'H':H}
        return Answer


def compute_objective_part(SPX,QQ0,Q0,Z,a,n,energy=False,subtract_mean=False,divide=1,L=None,R=None,L2=None,R2=None):
    # Divide by Z to normalize
    XiZ       = SPX[a:a+n]/Z
    QQ0iZ      = QQ0[a:a+n]/Z
    Q0iZ      = Q0[a:a+n]/Z
    # Subtract out the product of L and R if provided.
    if subtract_mean:
        if L is not None and R is not None:
            LiZ       = L[a:a+n]/Z
            RiZ       = R[a:a+n]/Z
            XiZ -= LiZ*RiZ
        elif L2 is not None and R2 is not None:
            L2iZ       = L2[a:a+n]/Z
            R2iZ       = R2[a:a+n]/Z
            XiZ -= L2iZ*R2iZ
        else:
            raise RuntimeError("subtract_mean is set to True, must provide L/R or L2/R2")
    if energy:
        QQ0iZ -= Q0iZ*Q0iZ

    # Return the answer.
    if divide == 1:
        X2      = np.sum(XiZ)/np.sum(QQ0iZ)
    elif divide == 0:
        X2      = np.sum(XiZ)
    elif divide == 2:
        X2      = np.sum(QQ0iZ)
    else:
        raise RuntimeError('Please pass either 0, 1, 2 to divide')
    return X2
