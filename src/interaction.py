""" @package interaction Interaction energy fitting module.

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

class Interaction(FittingSimulation):

    """ Subclass of FittingSimulation for fitting force fields to interaction energies.

    Currently Gromacs is supported.

    We introduce the following concepts:
    - The number of snapshots
    - The reference interaction energies and the file they belong in (qdata.txt)

    These will be taken care of at a later time:
    - The sampling simulation energies (emd0)
    - The WHAM Boltzmann weights (these are computed externally and passed in)
    - The QM Boltzmann weights (computed internally using the difference between eqm and emd0)

    There are also these little details:
    - Switches for whether to turn on certain Boltzmann weights (they stack)
    - Temperature for the QM Boltzmann weights

    This subclass contains the 'get' method for building the objective
    function from any simulation software (a driver to run the program and
    read output is still required)."""
    
    def __init__(self,options,sim_opts,forcefield):
        # Initialize the SuperClass!
        super(Interaction,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.ns            = sim_opts['shots']
        ## Whether to use WHAM Boltzmann weights
        self.whamboltz     = sim_opts['whamboltz']
        ## Whether to use QM Boltzmann weights
        self.qmboltz       = sim_opts['qmboltz']
        ## The temperature for QM Boltzmann weights
        self.qmboltztemp   = sim_opts['qmboltztemp']
        ## Whether to do energy and force calculations for the whole trajectory, or to do
        ## one calculation per snapshot.
        self.all_at_once   = sim_opts['all_at_once']
        ## OpenMM-only option - whether to run the energies and forces internally.
        self.run_internal  = sim_opts['run_internal']
        ## Do we call Q-Chem for dielectric energies?
        self.do_cosmo      = sim_opts['do_cosmo']
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## WHAM Boltzmann weights
        self.whamboltz_wts = []
        ## QM Boltzmann weights
        self.qmboltz_wts   = []
        ## Reference (QM) interaction energies
        self.eqm           = []
        ## Energies of the sampling simulation
        self.emd0          = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.simdir,"qdata.txt")
        ## Qualitative Indicator: average energy error (in kJ/mol)
        self.e_err = 0.0
        self.e_err_pct = None
        ## Read in the trajectory file
        if self.ns == -1:
            self.traj = Molecule(os.path.join(self.root,self.simdir,self.trajfnm))
            self.ns = len(self.traj)
        else:
            self.traj = Molecule(os.path.join(self.root,self.simdir,self.trajfnm))[:self.ns]
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)

    def read_reference_data(self):
        
        """ Read the reference ab initio data from a file such as qdata.txt.

        @todo Add an option for picking any slice out of
        qdata.txt, helpful for cross-validation
        
        @todo Closer integration of reference data with program -
        leave behind the qdata.txt format?  (For now, I like the
        readability of qdata.txt)

        After reading in the information from qdata.txt, it is converted
        into the GROMACS energy units.

        The configurations in force/energy matching typically come
        from a the thermodynamic ensemble of the MM force field at
        some temperature (by running MD, for example), and for many
        reasons it is helpful to introduce non-Boltzmann weights in
        front of these configurations.  There are two options: WHAM
        Boltzmann weights (for combining the weights of several
        simulations together) and QM Boltzmann weights (for converting
        MM weights into QM weights).  Note that the two sets of weights
        'stack'; i.e. they can be used at the same time.

        A 'hybrid' ensemble is possible where we use 50% MM and 50% QM
        weights.  Please read more in LPW and Troy Van Voorhis, JCP
        Vol. 133, Pg. 231101 (2010), doi:10.1063/1.3519043.
        
        @todo The WHAM Boltzmann weights are generated by external
        scripts (wanalyze.py and make-wham-data.sh) and passed in;
        perhaps these scripts can be added to the ForceBalance
        distribution or integrated more tightly.

        Finally, note that using non-Boltzmann weights degrades the
        statistical information content of the snapshots.  This
        problem will generally become worse if the ensemble to which
        we're reweighting is dramatically different from the one we're
        sampling from.  We end up with a set of Boltzmann weights like
        [1e-9, 1e-9, 1.0, 1e-9, 1e-9 ... ] and this is essentially just
        one snapshot.  I believe Troy is working on something to cure
        this problem.

        Here, we have a measure for the information content of our snapshots,
        which comes easily from the definition of information entropy:

        S = -1*Sum_i(P_i*log(P_i))
        InfoContent = exp(-S)

        With uniform weights, InfoContent is equal to the number of snapshots;
        with horrible weights, InfoContent is closer to one.

        """
        # Parse the qdata.txt file
        for line in open(os.path.join(self.root,self.qfnm)):
            sline = line.split()
            if len(sline) == 0: continue
            elif sline[0] == 'INTERACTION':
                self.eqm.append(float(sline[1]))
            if all(len(i) in [self.ns, 0] for i in [self.eqm]) and len(self.eqm) == self.ns:
                break
        self.ns = len(self.eqm)
        # Turn everything into arrays, convert to kJ/mol, and subtract the mean energy from the energy arrays
        self.eqm = array(self.eqm)
        self.eqm *= eqcgmx
        # Boltzmann weight stuff, copied from the energy / force matching section.
        self.emd0 = array(self.emd0)
        self.emd0 -= mean(self.emd0)
        if self.whamboltz == True:
            self.whamboltz_wts = array([float(i.strip()) for i in open(os.path.join(self.root,self.simdir,"wham-weights.txt")).readlines()])
            #   This is a constant pre-multiplier in front of every snapshot.
            bar = printcool("Using WHAM MM Boltzmann weights.", color=3)
            if os.path.exists(os.path.join(self.root,self.simdir,"wham-master.txt")):
                whaminfo = open(os.path.join(self.root,self.simdir,"wham-master.txt")).readlines()
                print "From wham-master.txt, I can see that you're using %i generations" % len(whaminfo)
                print "Relative weight of each generation:"
                shotcounter = 0
                for line in whaminfo:
                    sline = line.split()
                    genshots = int(sline[2])
                    weight = sum(self.whamboltz_wts[shotcounter:shotcounter+genshots])/sum(self.whamboltz_wts)
                    print " %s, %i snapshots, weight %.3e" % (sline[0], genshots, weight)
                    shotcounter += genshots
            else:
                print "Oops... WHAM files don't exist?"
            print bar
        else:
            self.whamboltz_wts = ones(self.ns)
        if self.qmboltz > 0:
            logboltz = -(self.eqm - mean(self.eqm) - self.emd0 + mean(self.emd0)) / kb / self.qmboltztemp
            logboltz -= max(logboltz) # Normalizes boltzmann weights
            self.qmboltz_wts = exp(logboltz)
            self.qmboltz_wts /= sum(self.qmboltz_wts)
            # where simply gathers the nonzero values otherwise we get lots of NaNs
            qbwhere = self.qmboltz_wts[where(self.qmboltz_wts)[0]]
            # Compute ze InfoContent!
            qmboltzent = -sum(qbwhere*log(qbwhere))
            print "Quantum Boltzmann weights are ON, the formula is exp(-b(E_qm-E_mm)),",
            print "distribution entropy is %.3f, equivalent to %.2f snapshots" % (qmboltzent, exp(qmboltzent))
            print "%.1f%% is mixed into the MM boltzmann weights." % (self.qmboltz*100)
        else:
            self.qmboltz_wts = ones(self.ns)

    def prepare_temp_directory(self, options, sim_opts):
        """ Prepare the temporary directory, by default does nothing """
        return
        
    def indicate(self):
        print "Sim: %-15s" % self.name, 
        print "Interaction Energy error = %8.4f kJ/mol (%.4f%%), Objective = %.5e" % (self.e_err, self.e_err_pct*100, self.objective)

    def get(self, mvals, AGrad=False, AHess=False):
        """
        LPW 05-30-2012
        
        This subroutine builds the objective function (and optionally
        its derivatives) from a general simulation software.  This is
        in contrast to using GROMACS-X2, which computes the objective
        function and prints it out; then 'get' only needs to call
        GROMACS and read it in.

        This subroutine interfaces with simulation software 'drivers'.
        The driver is only expected to give the interaction energy in kJ/mol.

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        """
        Answer = {}
        # Create the new force field!!
        pvals  = self.FF.make(mvals,self.usepvals)
        np     = self.FF.np
        ns     = self.ns
        #==============================================================#
        #            STEP 1: Form all of the arrays.                   #
        #                                                              #
        # An explanation of variable names follows.                    #
        # Z(Y)  = The partition function in the MM (QM) ensemble       #
        # QQ_Q  = The unnormalized expected value of <Q(X)Q>           #
        # Later on we divide these quantities by Z to normalize.       #
        #==============================================================#
        if (self.weight == 0.0):
            AGrad = False
            AHess = False
        Z       = 0
        Y       = 0
        Q       = 0
        # Derivatives
        M_p     = zeros(np,dtype=float)
        M_pp    = zeros(np,dtype=float)
        # Mean quantities
        Q0_M    = 0.0
        QQ_M    = 0.0
        Q0_Q    = 0.0
        QQ_Q    = 0.0
        # Objective functions
        SPiXi = 0
        SRiXi = 0
        if AGrad:
            SPiXi_p = zeros(np,dtype=float)
            SRiXi_p = zeros(np,dtype=float)
            X2_M_p = zeros(np,dtype=float)
            X2_Q_p = zeros(np,dtype=float)
        if AHess:
            SPiXi_pq = zeros((np,np),dtype=float)
            SRiXi_pq = zeros((np,np),dtype=float)
            X2_M_pq = zeros((np,np),dtype=float)
            X2_Q_pq = zeros((np,np),dtype=float)
        QBN = dot(self.qmboltz_wts[:ns],self.whamboltz_wts[:ns])
        if AGrad and self.all_at_once:
            dM_all = zeros((ns,np),dtype=float)
            ddM_all = zeros((ns,np),dtype=float)
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        interpids = ['VPAIR','COUL','VDW','POL']
        coulpids = ['COUL']
        if self.all_at_once:
            print "Executing\r",
            M_all = self.interaction_driver_all(dielectric=self.do_cosmo)
            if AGrad or AHess:
                def callM(mvals_, dielectric=False):
                    print "\r",
                    pvals = self.FF.make(mvals_, self.usepvals)
                    return self.interaction_driver_all(dielectric)
                for p in range(np):
                    if any([j in self.FF.plist[p] for j in interpids]):
                        # Differentiate only if the parameter is relevant for intermolecular interactions. :)
                        #dM_all[:,p], ddM_all[:,p] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M_all)
                        dM_all[:,p] = f1d2p(fdwrap(callM, mvals, p, dielectric=self.do_cosmo and any([j in self.FF.plist[p] for j in coulpids])), h = self.h, f0 = M_all)
            # Dump interaction energies to disk.
            savetxt('M.txt',M_all)
            savetxt('Q.txt',self.eqm)
        for i in range(ns):
            print "Incrementing quantities for snapshot %i\r" % i,
            # Build Boltzmann weights and increment partition function.
            P   = self.whamboltz_wts[i]
            Z  += P
            R   = self.qmboltz_wts[i]*self.whamboltz_wts[i] / QBN
            Y  += R
            # Recall reference (QM) data
            Q   = self.eqm[i]
            QQ  = Q*Q
            # Call the simulation software to get the MM quantities.
            if self.all_at_once:
                M = M_all[i]
            else:
                print "Shot %i\r" % i,
                M = self.interaction_driver(i)
            X     = M-Q
            # Increment the average values.
            Q0_M += P*Q
            Q0_Q += R*Q
            QQ_M += P*QQ
            QQ_Q += R*QQ
            # Increment the objective function.
            Xi     = X**2
            SPiXi += P * Xi
            SRiXi += R * Xi
            for p in range(np):
                if not AGrad: continue
                if self.all_at_once:
                    M_p[p] = dM_all[i, p]
                    M_pp[p] = ddM_all[i, p]
                else:
                    def callM(mvals_):
                        print "\r",
                        pvals = self.FF.make(mvals_, self.usepvals)
                        return self.energy_force_driver(i)
                    M_p[p] = f1d2p(fdwrap(callM, mvals, p), h = self.h)
                Xi_p        = 2 * X * M_p[p]
                SPiXi_p[p] += P * Xi_p
                SRiXi_p[p] += R * Xi_p
                if not AHess: continue
                Xi_pq       = 2 * (M_p[p] * M_p[p] + X * M_pp[p])
                SPiXi_pq[p,p] += P * Xi_pq
                SRiXi_pq[p,p] += R * Xi_pq
                for q in range(p):
                    Xi_pq          = 2 * M_p[p] * M_p[q]
                    SPiXi_pq[p,q] += P * Xi_pq
                    SRiXi_pq[p,q] += R * Xi_pq
        #==============================================================#
        #      STEP 3: Build the variance vector and invert it.        #
        #==============================================================#
        print "Done with snapshots, building objective function now\r",
        # Here we're just using the variance.
        QBP  = self.qmboltz
        MBP  = 1 - self.qmboltz
        C    = MBP*(QQ_M-Q0_M*Q0_M/Z)/Z + QBP*(QQ_Q-Q0_Q*Q0_Q/Y)/Y
        Ci   = 1. / C
        #==============================================================#
        # STEP 4: Build the objective function and its derivatives.    #
        #==============================================================#
        X2_M  = SPiXi*Ci/Z
        X2_Q  = SRiXi*Ci/Y
        for p in range(np):
            if not AGrad: continue
            X2_M_p[p] = SPiXi_p[p]/C/Z
            X2_Q_p[p] = SRiXi_p[p]/C/Y
            if not AHess: continue
            X2_M_pq[p,p] = SPiXi_pq[p,p]/C/Z
            X2_Q_pq[p,p] = SRiXi_pq[p,p]/C/Y
            for q in range(p):
                X2_M_pq[p,q] = SPiXi_pq[p,q]/C/Z
                X2_Q_pq[p,q] = SRiXi_pq[p,q]/C/Y
                # Get the other half of the Hessian matrix.
                X2_M_pq[q,p] = X2_M_pq[p,q]
                X2_Q_pq[q,p] = X2_Q_pq[p,q]
                
        #==============================================================#
        #            STEP 5: Build the return quantities.              #
        #==============================================================#
        # The objective function
        X2   = MBP * X2_M    + QBP * X2_Q
        # Derivatives of the objective function
        G = zeros(np,dtype=float)
        H = zeros((np,np),dtype=float)
        for p in range(np):
            if not AGrad: continue
            G[p] = MBP * X2_M_p[p] + QBP * X2_Q_p[p]
            if not AHess: continue
            for q in range(np):
                H[p,q] = MBP * X2_M_pq[p,q] + QBP * X2_Q_pq[p,q]
        # Energy error in kJ/mol
        E     = MBP * sqrt(SPiXi/Z) + QBP * sqrt(SRiXi/Y)
        # Fractional energy error.
        Efrac = MBP * sqrt((SPiXi/Z) / (QQ_M/Z - Q0_M**2/Z/Z)) + QBP * sqrt((SRiXi/Y) / (QQ_Q/Y - Q0_Q**2/Y/Y))
        self.e_err = E
        self.e_err_pct = Efrac
        Answer = {'X':X2, 'G':G, 'H':H}
        if not in_fd():
            self.objective = Answer['X']
        return Answer
