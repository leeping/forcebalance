""" @package abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, warn_press_key
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt
from fitsim import FittingSimulation
from molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from finite_difference import fdwrap, f1d2p, f12d3p, in_fd
#from _increment import AbInitio_Build

class AbInitio(FittingSimulation):

    """ Subclass of FittingSimulation for fitting force fields to ab initio data.

    Currently Gromacs-X2, Gromacs, Tinker, OpenMM and AMBER are supported.

    We introduce the following concepts:
    - The number of snapshots
    - The reference energies and forces (eqm, fqm) and the file they belong in (qdata.txt)
    - The sampling simulation energies (emd0)
    - The WHAM Boltzmann weights (these are computed externally and passed in)
    - The QM Boltzmann weights (computed internally using the difference between eqm and emd0)

    There are also these little details:
    - Switches for whether to turn on certain Boltzmann weights (they stack)
    - Temperature for the QM Boltzmann weights
    - Whether to fit a subset of atoms

    This subclass contains the 'get' method for building the objective
    function from any simulation software (a driver to run the program and
    read output is still required).  The 'get' method can be overridden
    by subclasses like AbInitio_GMXX2."""
    
    def __init__(self,options,sim_opts,forcefield):
        """Instantiation of the subclass.

        We begin by instantiating the superclass here and also
        defining a number of core concepts for energy / force
        matching.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """
        
        # Initialize the SuperClass!
        super(AbInitio,self).__init__(options,sim_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.ns            = sim_opts['shots']
        ## Whether to use WHAM Boltzmann weights
        self.whamboltz     = sim_opts['whamboltz']
        ## Whether to use the Sampling Correction
        self.sampcorr      = sim_opts['sampcorr']
        ## Whether to use the Covariance Matrix
        self.covariance    = sim_opts['covariance']
        ## Whether to use QM Boltzmann weights
        self.qmboltz       = sim_opts['qmboltz']
        ## The temperature for QM Boltzmann weights
        self.qmboltztemp   = sim_opts['qmboltztemp']
        ## Number of atoms that we are fitting
        self.fitatoms      = sim_opts['fitatoms']
        ## Whether to fit Energies.
        self.energy        = sim_opts['energy']
        ## Whether to fit Forces.
        self.force         = sim_opts['force']
        ## Whether to fit Electrostatic Potential.
        self.resp          = sim_opts['resp']
        self.resp_a        = sim_opts['resp_a']
        self.resp_b        = sim_opts['resp_b']
        ## Weights for the three components.
        self.w_energy      = sim_opts['w_energy']
        self.w_force       = sim_opts['w_force']
        self.w_resp        = sim_opts['w_resp']
        ## Whether to do energy and force calculations for the whole trajectory, or to do
        ## one calculation per snapshot.
        self.all_at_once   = sim_opts['all_at_once']
        ## OpenMM-only option - whether to run the energies and forces internally.
        self.run_internal  = sim_opts['run_internal']
        ## Whether we have virtual sites (set at the global option level)
        self.have_vsite    = options['have_vsite']
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## WHAM Boltzmann weights
        self.whamboltz_wts = []
        ## QM Boltzmann weights
        self.qmboltz_wts   = []
        ## Reference (QM) energies
        self.eqm           = []
        ## Energies of the sampling simulation
        self.emd0          = []
        ## Reference (QM) forces
        self.fqm           = []
        ## ESP grid points
        self.espxyz        = []
        ## ESP values
        self.espval        = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.simdir,"qdata.txt")
        ## The number of true atoms 
        self.natoms      = 0
        ## Qualitative Indicator: average energy error (in kJ/mol)
        self.e_err = 0.0
        self.e_err_pct = None
        ## Qualitative Indicator: average force error (fractional)
        self.f_err = 0.0
        ## Qualitative Indicator: "relative RMS" for electrostatic potential
        self.esp_err = 0.0
        ## Read in the trajectory file
        if self.ns == -1:
            self.traj = Molecule(os.path.join(self.root,self.simdir,self.trajfnm))
            self.ns = len(self.traj)
        else:
            self.traj = Molecule(os.path.join(self.root,self.simdir,self.trajfnm))[:self.ns]
        ## The number of (atoms + drude particles + virtual sites)
        self.nparticles  = len(self.traj.elem)
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,sim_opts)
        ## The below two options are related to whether we want to rebuild virtual site positions.
        ## Rebuild the distance matrix if virtual site positions have changed
        self.new_vsites = True
        ## Save the mvals from the last time we updated the vsites.
        self.save_vmvals = {}

    def build_invdist(self, mvals):
        for i in range(self.FF.np):
            if 'VSITE' in self.FF.plist[i]:
                if i in self.save_vmvals and mvals[i] != self.save_vmvals[i]:
                    self.new_vsites = True
                    break
        if not self.new_vsites: return self.invdists
        if any(['VSITE' in i for i in self.FF.map.keys()]) or self.have_vsite:
            print "\rGenerating virtual site positions.%s" % (" "*30),
            pvals = self.FF.make(mvals,self.usepvals)
            self.generate_vsite_positions()
        # prepare the distance matrix for esp computations
        if len(self.espxyz) > 0:
            invdists = []
            print "\rPreparing the distance matrix... it will have %i * %i * %i = %i elements" % (self.ns, self.nesp, self.nparticles, self.ns * self.nesp * self.nparticles),
            sn = 0
            for espset, xyz in zip(self.espxyz, self.traj.xyzs):
                print "\rGenerating ESP distances for snapshot %i%s" % (sn, " "*50),
                esparr = array(espset).reshape(-1,3)
                # Create a matrix with Nesp rows and Natoms columns.
                DistMat = array([[linalg.norm(i - j) for j in xyz] for i in esparr])
                #print "DistMat min/max: % .3f % .3f" % (min(flat(DistMat)), max(flat(DistMat)))
                # xyztest = ['%i' % (self.nesp + self.nparticles),'Testing ESP for frame number %i' % sn]
                # for i, x in enumerate(xyz):
                #     xyztest.append(format_xyz_coord(sub('[0-9]','',self.FF.atomnames[i]),x))
                # for i in esparr:
                #     xyztest.append(format_xyz_coord("He",i))
                #with open('test.xyz','w' if sn == 0 else 'a') as f: f.writelines([l+'\n' for l in xyztest])
                invdists.append(1. / (DistMat / bohrang))
                sn += 1
        for i in range(self.FF.np):
            if 'VSITE' in self.FF.plist[i]:
                self.save_vmvals[i] = mvals[i]
        self.new_vsites = False
        return array(invdists)

    def read_reference_data(self):
        
        """ Read the reference ab initio data from a file such as qdata.txt.

        @todo Add an option for picking any slice out of
        qdata.txt, helpful for cross-validation
        
        @todo Closer integration of reference data with program -
        leave behind the qdata.txt format?  (For now, I like the
        readability of qdata.txt)

        After reading in the information from qdata.txt, it is converted
        into the GROMACS energy units (kind of an arbitrary choice);
        forces (kind of a misnomer in qdata.txt) are multipled by -1
        to convert gradients to forces.

        We also subtract out the mean energies of all energy arrays
        because energy/force matching does not account for zero-point
        energy differences between MM and QM (i.e. energy of electrons
        in core orbitals).

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
            elif sline[0] == 'ENERGY':
                self.eqm.append(float(sline[1]))
            elif sline[0] == 'EMD0':
                self.emd0.append(float(sline[1]))
            elif sline[0] == 'FORCES':
                self.fqm.append([float(i) for i in sline[1:]])
            elif sline[0] == 'ESPXYZ':
                self.espxyz.append([float(i) for i in sline[1:]])
            elif sline[0] == 'ESPVAL':
                self.espval.append([float(i) for i in sline[1:]])
            if all(len(i) in [self.ns, 0] for i in [self.eqm, self.fqm, self.emd0, self.espxyz, self.espval]) and len(self.eqm) == self.ns:
                break
        self.ns = len(self.eqm)
        # Turn everything into arrays, convert to kJ/mol, and subtract the mean energy from the energy arrays
        self.eqm = array(self.eqm)
        self.eqm *= eqcgmx
        self.eqm -= mean(self.eqm)
        self.fqm = array(self.fqm)
        self.fqm *= fqcgmx
        self.natoms = self.fqm.shape[1]/3
        self.nesp = len(self.espval[0]) if len(self.espval) > 0 else 0
        # Here we may choose a subset of atoms to do the force matching.
        if self.fitatoms == 0:
            self.fitatoms = self.natoms
        elif self.fitatoms > self.natoms:
            print "What the heck, more fitting atoms than the total number of atoms?"
            sys.exit(1)
        else:
            # Indicate to Gromacs that we're only fitting the first however-many atoms.
            print "We're only fitting the first %i atoms" % self.fitatoms
            print "The quantum force matrix appears to contain more components (%i) than those being fit (%i)." % (fqmm.shape[1], 3*self.fitatoms)
            print "Pruning the quantum force matrix..."
            self.fqm  = self.fqm[:, :3*self.fitatoms].copy()
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
            # LPW I haven't revised this section yet
            # Do I need to?
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
        """ Prepare the temporary directory, by default does nothing (gmxx2 needs it) """
        return
        
    def indicate(self):
        print "Sim: %-15s" % self.name, 
        if self.energy or self.force:
            if self.e_err_pct == None:
                print "Energy error (kJ/mol) = %8.4f Force error (%%) = %8.4f" % (self.e_err, self.f_err*100), 
            else:
                print "Energy error = %8.4f kJ/mol (%.4f%%) Force error (%%) = %8.4f" % (self.e_err, self.e_err_pct*100, self.f_err*100), 
        if self.resp:
            print "ESP_err (%%) = %8.4f, RESP penalty = %.3e" % (self.esp_err*100, self.respterm),
        print "Objective = %.5e" % self.objective

    def get_energy_force_no_covariance_(self, mvals, AGrad=False, AHess=False):
        """
        LPW 05-30-2012
        
        This subroutine builds the objective function (and optionally
        its derivatives) from a general simulation software.  This is
        in contrast to using GROMACS-X2, which computes the objective
        function and prints it out; then 'get' only needs to call
        GROMACS and read it in.

        This subroutine interfaces with simulation software 'drivers'.
        The driver is only expected to give the energy and forces.

        Now this subroutine may sound trivial since the objective
        function is simply a least-squares quantity (M-Q)^2 - but
        there are a number of nontrivial considerations.  I will list
        them here.
        
        0) Polytensor formulation: Removed because it adds a factor of
        NCP1 to the memory requirement.  Instead we're trying
        Gauss-Newton approximation to the Hessian.

        1) Boltzmann weights and normalization: Each snapshot has its
        own Boltzmann weight, which may or may not be normalized.
        This subroutine does the normalization automatically.
        
        2) Subtracting out the mean energy gap: The zero-point energy
        difference between reference and fitting simulations is
        meaningless.  This subroutine subtracts it out.
        
        3) Hybrid ensembles: This program builds a combined objective
        function from both MM and QM ensembles, which is rigorously
        better than using a single ensemble.

        Note that this subroutine does not do EVERYTHING that
        GROMACS-X2 can do, which includes:
        
        1) Internal coordinate systems
        2) 'Sampling correction' (deprecated, since it doesn't seem to work)
        3) Analytic derivatives
        
        @todo Parallelization over snapshots is not implemented yet

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        """
        Answer = {}
        # Create the new force field!!
        pvals = self.FF.make(mvals,self.usepvals)

        #======================================#
        #   Copied from the old ForTune code   #
        #======================================#
        NC   = 3*self.fitatoms
        NCP1 = 3*self.fitatoms+1
        np   = self.FF.np
        ns   = self.ns
        #==============================================================#
        # Note: Because of hybrid ensembles, we form two separate      #
        # objective functions.  This means the code is (trivially)     #
        # doubled in length because there are two sets of Boltzmann    #
        # weights.  In principle I could write a general subroutine    #
        # for a single set of Boltzmann weights and call it twice, but #
        # that would require me to loop over the snapshots twice.      #
        # However, after the loop over snapshots, the subroutine that  #
        # builds the objective function (build_objective, above) is    #
        # called twice using the MM-ensemble and QM-ensemble           #
        # variables.                                                   #
        #                                                              #
        #            STEP 1: Form all of the arrays.                   #
        #                                                              #
        # An explanation of variable names follows.                    #
        # Z(Y)  = The partition function in the MM (QM) ensemble       #
        # M0_M  = The mean of the MM values in the MM ensemble         #
        # QQ_Q  = The unnormalized expected value of <Q(X)Q>           #
        # Later on we divide these quantities by Z to normalize.       #
        #==============================================================#
        if (self.w_energy == 0.0 and self.w_force == 0.0):
            AGrad = False
            AHess = False
        Z       = 0.0
        Y       = 0.0
        Q = zeros(NCP1,dtype=float)
        # Derivatives
        M_p     = zeros((np,NCP1),dtype=float)
        M_pp    = zeros((np,NCP1),dtype=float)
        # Mean quantities
        M0_M    = zeros(NCP1,dtype=float)
        X0_M    = zeros(NCP1,dtype=float)
        Q0_M    = zeros(NCP1,dtype=float)
        QQ_M    = zeros(NCP1,dtype=float)
        M0_Q    = zeros(NCP1,dtype=float)
        X0_Q    = zeros(NCP1,dtype=float)
        Q0_Q    = zeros(NCP1,dtype=float)
        QQ_Q    = zeros(NCP1,dtype=float)
        # Means of gradients
        M0_M_p  = zeros((np,NCP1),dtype=float)
        M0_Q_p  = zeros((np,NCP1),dtype=float)
        M0_M_pp = zeros((np,NCP1),dtype=float)
        M0_Q_pp = zeros((np,NCP1),dtype=float)
        # Objective functions
        SPiXi = zeros(NCP1,dtype=float)
        SRiXi = zeros(NCP1,dtype=float)
        if AGrad:
            SPiXi_p = zeros((np,NCP1),dtype=float)
            SRiXi_p = zeros((np,NCP1),dtype=float)
            X2_M_p = zeros(np,dtype=float)
            X2_Q_p = zeros(np,dtype=float)
        if AHess:
            SPiXi_pq = zeros((np,np,NCP1),dtype=float)
            SRiXi_pq = zeros((np,np,NCP1),dtype=float)
            X2_M_pq = zeros((np,np),dtype=float)
            X2_Q_pq = zeros((np,np),dtype=float)
        QBN = dot(self.qmboltz_wts[:ns],self.whamboltz_wts[:ns])
        if AGrad and self.all_at_once:
            dM_all = zeros((ns,np,NCP1),dtype=float)
            ddM_all = zeros((ns,np,NCP1),dtype=float)
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        if self.all_at_once:
            print "\rExecuting\r",
            M_all = self.energy_force_driver_all()
            if AGrad or AHess:
                def callM(mvals_):
                    print "\r",
                    pvals = self.FF.make(mvals_, self.usepvals)
                    return self.energy_force_driver_all()
                for p in range(np):
                    dM_all[:,p,:], ddM_all[:,p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M_all)
            # Dump energies and forces to disk.
            savetxt('M.txt',M_all)
        # My C helper code isn't fully functional yet.
        # try:
        #     AbInitio_Build(np, ns, NCP1, AGrad, AHess,
        #                    self.whamboltz_wts, self.qmboltz_wts, QBN, Z, Y, self.eqm, self.fqm, M_all, 
        #                    X0_M, M0_M, Q0_M, QQ_M, X0_Q, M0_Q, Q0_Q, QQ_Q, 
        #                    SPiXi, SRiXi, dM_all, ddM_all, M0_M_p, M0_Q_p, 
        #                    M0_M_pp, M0_Q_pp, SPiXi_p, SRiXi_p, SPiXi_pq, SRiXi_pq)
        # except:
        #     warn_press_key("AbInitio_Build has phailed!")
        for i in range(ns):
            print "\rIncrementing quantities for snapshot %i\r" % i,
            # Build Boltzmann weights and increment partition function.
            P   = self.whamboltz_wts[i]
            Z  += P
            R   = self.qmboltz_wts[i]*self.whamboltz_wts[i] / QBN
            Y  += R
            # Recall reference (QM) data
            Q[0] = self.eqm[i]
            Q[1:] = self.fqm[i,:].copy()
            QQ     = Q*Q
            # Call the simulation software to get the MM quantities.
            if self.all_at_once:
                M = M_all[i]
            else:
                print "Shot %i\r" % i,
                M = self.energy_force_driver(i)
            X     = M-Q
            # Increment the average values.
            X0_M += P*X
            X0_Q += R*X
            M0_M += P*M
            M0_Q += R*M
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
                    M_p[p],M_pp[p] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
                M0_M_p[p]  += P * M_p[p]
                M0_Q_p[p]  += R * M_p[p]
                #M0_M_pp[p] += P * M_pp[p]
                #M0_Q_pp[p] += R * M_pp[p]
                Xi_p        = 2 * X * M_p[p]
                SPiXi_p[p] += P * Xi_p
                SRiXi_p[p] += R * Xi_p
                if not AHess: continue
                M_pp[p] = ddM_all[i, p]
                # This formula is more correct, but perhapsively convergence is slower.
                #Xi_pq       = 2 * (M_p[p] * M_p[p] + X * M_pp[p])
                # Gauss-Newton formula for approximate Hessian
                Xi_pq       = 2 * (M_p[p] * M_p[p])
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
        if (self.w_energy > 0.0 and self.w_force > 0.0):
            EFW     = self.w_energy / (self.w_energy + self.w_force)
            CEFW    = 1.0 - EFW
        else:
            EFW = 0.0
            CEFW = 0.0
        # Build the weight vector, so the force contribution is suppressed by 1/3N
        WM      = zeros(NCP1,dtype=float)
        WM[0] = sqrt(EFW)
        for i in range(1,NCP1):
            WM[i] = sqrt(CEFW/NC)
        # Here we're just using the variance.
        QBP  = self.qmboltz
        MBP  = 1 - self.qmboltz
        C    = MBP*(QQ_M-Q0_M*Q0_M/Z)/Z + QBP*(QQ_Q-Q0_Q*Q0_Q/Y)/Y
        # Normalize all of the force components (should I do this?)
        C[1:] = mean(C[1:])
        Ci    = 1. / C
        WCiW = WM * Ci * WM
        #==============================================================#
        # STEP 4: Build the objective function and its derivatives.    #
        #==============================================================#
        X2_M  = weighted_variance(SPiXi,WCiW,Z,X0_M,X0_M,NCP1)
        X2_Q  = weighted_variance(SRiXi,WCiW,Y,X0_Q,X0_Q,NCP1)
        for p in range(np):
            if not AGrad: continue
            X2_M_p[p] = weighted_variance(SPiXi_p[p],WCiW,Z,2*X0_M,M0_M_p[p],NCP1)
            X2_Q_p[p] = weighted_variance(SRiXi_p[p],WCiW,Y,2*X0_Q,M0_Q_p[p],NCP1)
            if not AHess: continue
            X2_M_pq[p,p] = weighted_variance2(SPiXi_pq[p,p],WCiW,Z,2*M0_M_p[p],M0_M_p[p],2*X0_M,M0_M_pp[p],NCP1)
            X2_Q_pq[p,p] = weighted_variance2(SRiXi_pq[p,p],WCiW,Y,2*M0_Q_p[p],M0_Q_p[p],2*X0_Q,M0_Q_pp[p],NCP1)
            for q in range(p):
                X2_M_pq[p,q] = weighted_variance(SPiXi_pq[p,q],WCiW,Z,2*M0_M_p[p],M0_M_p[q],NCP1)
                X2_Q_pq[p,q] = weighted_variance(SRiXi_pq[p,q],WCiW,Y,2*M0_Q_p[p],M0_Q_p[q],NCP1)
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
        E0_M = (2*Q0_M[0]*M0_M[0] - Q0_M[0]*Q0_M[0] - M0_M[0]*M0_M[0])/Z/Z;
        E0_Q = (2*Q0_Q[0]*M0_Q[0] - Q0_Q[0]*Q0_Q[0] - M0_Q[0]*M0_Q[0])/Y/Y;
        E     = MBP * sqrt(SPiXi[0]/Z + E0_M) + QBP * sqrt(SRiXi[0]/Y + E0_Q)
        # Fractional energy error.
        Efrac = MBP * sqrt((SPiXi[0]/Z + E0_M) / (QQ_M[0]/Z - Q0_M[0]**2/Z/Z)) + QBP * sqrt((SRiXi[0]/Y + E0_Q) / (QQ_Q[0]/Y - Q0_Q[0]**2/Y/Y))
        # Fractional force error.
        F     = MBP * sqrt(mean(array([SPiXi[i]/QQ_M[i] for i in range(1,NCP1)]))) + \
                QBP * sqrt(mean(array([SRiXi[i]/QQ_Q[i] for i in range(1,NCP1)])))
        if not in_fd():
            self.e_err = E
            self.e_err_pct = Efrac
            self.f_err = F
        Answer = {'X':X2, 'G':G, 'H':H}
        return Answer

    def get_energy_force_covariance_(self, mvals, AGrad=False, AHess=False):
        """
        LPW 01-11-2012
        
        This subroutine builds the objective function (and optionally
        its derivatives) from a general simulation software.  This is
        in contrast to using GROMACS-X2, which computes the objective
        function and prints it out; then 'get' only needs to call
        GROMACS and read it in.

        This subroutine interfaces with simulation software 'drivers'.
        The driver is only expected to give the energy and forces.

        Now this subroutine may sound trivial since the objective
        function is simply a least-squares quantity (M-Q)^2 - but
        there are a number of nontrivial considerations.  I will list
        them here.
        
        0) Polytensor formulation: Because there may exist covariance
        between different components of the force (or covariance
        between the energy and the force), we build the objective
        function by taking outer products of vectors that have the
        form [E F_1x F_1y F_1z F_2x F_2y ... ], and then we trace it
        with the inverse of the covariance matrix to get the objective
        function.

        1) Boltzmann weights and normalization: Each snapshot has its
        own Boltzmann weight, which may or may not be normalized.
        This subroutine does the normalization automatically.
        
        2) Subtracting out the mean energy gap: The zero-point energy
        difference between reference and fitting simulations is
        meaningless.  This subroutine subtracts it out.
        
        3) Hybrid ensembles: This program builds a combined objective
        function from both MM and QM ensembles, which is rigorously
        better than using a single ensemble.

        Note that this subroutine does not do EVERYTHING that
        GROMACS-X2 can do, which includes:
        
        1) Internal coordinate systems
        2) 'Sampling correction' (deprecated, since it doesn't seem to work)
        3) Analytic derivatives
        
        In the previous code (ForTune) this subroutine used analytic
        first derivatives of the energy and force to build the
        derivatives of the objective function.  Here I will take a
        simplified approach, because building the derivatives are
        cumbersome.  For now we will return the objective function
        ONLY.  A two-point central difference should give us the first
        and diagonal second derivative anyhow.

        @todo Parallelization over snapshots is not implemented yet

        @param[in] mvals Mathematical parameter values
        @param[in] AGrad Switch to turn on analytic gradient
        @param[in] AHess Switch to turn on analytic Hessian
        @return Answer Contribution to the objective function
        """
        Answer = {}
        # Create the new force field!!
        pvals = self.FF.make(mvals,self.usepvals)
        ns = self.ns

        #======================================#
        #   Copied from the old ForTune code   #
        #======================================#
        NC   = 3*self.fitatoms
        NCP1 = 3*self.fitatoms+1
        #==============================================================#
        # Note: Because of hybrid ensembles, we form two separate      #
        # objective functions.  This means the code is (trivially)     #
        # doubled in length because there are two sets of Boltzmann    #
        # weights.  In principle I could write a general subroutine    #
        # for a single set of Boltzmann weights and call it twice, but #
        # that would require me to loop over the snapshots twice.      #
        # However, after the loop over snapshots, the subroutine that  #
        # builds the objective function (build_objective, above) is    #
        # called twice using the MM-ensemble and QM-ensemble           #
        # variables.                                                   #
        #                                                              #
        #            STEP 1: Form all of the arrays.                   #
        #                                                              #
        # Quantities pertaining to the covariance matrix.              #
        # An explanation of variable names follows.                    #
        # Z(Y)  = The partition function in the MM (QM) ensemble       #
        # M0_M  = The mean of the MM values in the MM ensemble         #
        # QQ_Q  = The unnormalized expected value of <Q(X)Q>           #
        # Later on we divide these quantities by Z to normalize,       #
        # and Q0(X)Q0 is subtracted from QQ to get the covariance.     #
        #==============================================================#
        if (self.w_energy == 0.0 and self.w_force == 0.0):
            AGrad = False
            AHess = False
        Z       = 0
        Y       = 0
        Q = zeros(NCP1,dtype=float)
        M0_M    = zeros(NCP1,dtype=float)
        Q0_M    = zeros(NCP1,dtype=float)
        QQ_M    = zeros((NCP1,NCP1),dtype=float)
        M0_Q    = zeros(NCP1,dtype=float)
        Q0_Q    = zeros(NCP1,dtype=float)
        QQ_Q    = zeros((NCP1,NCP1),dtype=float)
        #==============================================================#
        # Objective function polytensors: This is formed in a loop     #
        # over snapshots by taking the outer product (Q-M)(X)(Q-M),    #
        # multiplying by the Boltzmann weight, and then summing.       #
        #==============================================================#
        SPiXi = zeros((NCP1,NCP1),dtype=float)
        SRiXi = zeros((NCP1,NCP1),dtype=float)
        QBN = dot(self.qmboltz_wts[:ns],self.whamboltz_wts[:ns])
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        if self.all_at_once:
            print "Computing forces\r",
            M_all = self.energy_force_driver_all()
        for i in range(ns):
            # Build Boltzmann weights and increment partition function.
            P   = self.whamboltz_wts[i]
            Z  += P
            R   = self.qmboltz_wts[i]*self.whamboltz_wts[i] / QBN
            Y  += R
            # Recall reference (QM) data
            Q[0] = self.eqm[i]
            Q[1:] = self.fqm[i,:].copy()
            # Increment the average quantities.
            QQ     = outer(Q,Q)
            # Call the simulation software to get the MM quantities.
            if self.all_at_once:
                M = M_all[i]
            else:
                print "Shot %i\r" % i,
                M = self.energy_force_driver(i)
            # Increment the average values.
            M0_M += P*M
            Q0_M += P*Q
            QQ_M += P*QQ
            M0_Q += R*M
            Q0_Q += R*Q
            QQ_Q += R*QQ
            # Increment the objective function.
            Xi  = outer(M,M) - 2*outer(Q,M) + outer(Q,Q)
            SPiXi += P * Xi
            SRiXi += R * Xi
        #==============================================================#
        #     STEP 3: Build the covariance matrix and invert it.       #
        #==============================================================#
        if (self.w_energy > 0.0 and self.w_force > 0.0):
            EFW     = self.w_energy / (self.w_energy + self.w_force)
            CEFW    = 1.0 - EFW
        else:
            EFW = 0.0
            CEFW = 0.0
        # Build the weight matrix, so the force contribution is suppressed by 1/3N
        WM      = zeros((NCP1,NCP1),dtype=float)
        WM[0,0] = sqrt(EFW)
        for i in range(1,NCP1):
            WM[i,i] = sqrt(CEFW/NC)
        #==============================================================#
        #                      The covariance matrix                   #
        #                                                              #
        # Note: There is a detail here that might appear peculiar :    #
        # I'm building only one covariance matrix from both ensembles. #
        # This is the proper thing to do, believe me.                  #
        # The functional form looks like (X2_M + X2_Q)(C)^-1           #
        # I will build X2_M C^-1 and X2_Q C^-1 separately.             #
        #==============================================================#
        QBP  = self.qmboltz
        MBP  = 1 - self.qmboltz
        C    = MBP*(QQ_M-outer(Q0_M,Q0_M)/Z)/Z + QBP*(QQ_Q-outer(Q0_Q,Q0_Q)/Y)/Y
        if self.covariance == False:
            C = diag(C)
            C[1:] = mean(C[1:])
            C = diag(C)
        Ci   = invert_svd(C)
        # Get rid of energy-force covariance
        for i in range(1,NCP1):
            Ci[0,i] = 0.0;
            Ci[i,0] = 0.0;
        WCiW = array(mat(WM) * mat(Ci) * mat(WM)) # Weighted covariance matrix.
        #==============================================================#
        # STEP 4: Build the objective function and its derivatives by  #
        # tracing with the covariance matrix.  (Note that without      #
        # Sampling Correction, covariance matrix has no derivatives.   #
        # I am thankful for this.)  I will build X2_M C^-1 and X2_Q    #
        # C^-1 separately because it's cleaner that way.               #
        #==============================================================#
        X2_M  = build_objective(SPiXi,WCiW,Z,Q0_M,M0_M,NCP1)
        X2_Q  = build_objective(SRiXi,WCiW,Y,Q0_Q,M0_Q,NCP1)
        #==============================================================#
        #            STEP 5: Build the return quantities.              #
        #==============================================================#
        # The un-penalized objective function
        BC   = MBP * X2_M    + QBP * X2_Q
        # Energy error in kJ/mol
        E0_M = (2*Q0_M[0]*M0_M[0] - Q0_M[0]*Q0_M[0] - M0_M[0]*M0_M[0])/Z/Z;
        E0_Q = (2*Q0_Q[0]*M0_Q[0] - Q0_Q[0]*Q0_Q[0] - M0_Q[0]*M0_Q[0])/Y/Y;
        E     = MBP * sqrt(SPiXi[0,0]/Z + E0_M) + QBP * sqrt(SRiXi[0,0]/Y + E0_Q)
        #==============================================================#
        # Fractional energy error.  This is not used right now but I   #
        # thought it might help to have the formula.  Note: SPiXi and  #
        # QQ both need to be divided by Z to be truly "averaged", but  #
        # since we're dividing them, the Z cancels out.                #
        #==============================================================#
        Efrac = MBP * sqrt((SPiXi[0]/Z + E0_M) / (QQ_M[0]/Z - Q0_M[0]**2/Z/Z)) + QBP * sqrt((SRiXi[0]/Y + E0_Q) / (QQ_Q[0]/Y - Q0_Q[0]**2/Y/Y))
        #==============================================================#
        # Fractional force error; this is the internal force error,    #
        # and for us this only means it is Boltzmann-averaged, because #
        # we are not doing the sampling correction or the internal     #
        # coordinates.  This is given by the root-mean-square diagonal #
        # objective function values (divided by the diagonal           #
        # covariance values).                                          #
        #==============================================================#
        F     = MBP * sqrt(mean(array([SPiXi[i,i]/QQ_M[i,i] for i in range(1,NCP1)]))) + \
                QBP * sqrt(mean(array([SRiXi[i,i]/QQ_Q[i,i] for i in range(1,NCP1)])))
        #======================================#
        #        End of the copied code        #
        #======================================#
        if not in_fd():
            self.e_err = E
            self.e_err_pct = Efrac
            self.f_err = F
        Answer = {'X':BC, 'G':zeros(self.FF.np), 'H':zeros((self.FF.np,self.FF.np))}
        return Answer

    def get_resp_(self, mvals, AGrad=False, AHess=False):
        """ Electrostatic potential fitting.  Implements the RESP objective function.  (In Python so obviously not optimized.) """
        if (self.w_resp == 0.0):
            AGrad = False
            AHess = False
        Answer = {}
        pvals = self.FF.make(mvals,self.usepvals)

        # Build the distance matrix for ESP fitting.
        self.invdists = self.build_invdist(mvals)

        ns = self.ns
        np = self.FF.np
        Z = 0
        Y = 0
        def getqatoms(mvals_):
            """ This function takes the mathematical parameter values and returns the charges on the ATOMS (fancy mapping going on) """
            print "\r",
            # Need to update the positions of atoms, if there are virtual sites.
            pvals = self.FF.create_pvals(mvals_)
            qvals = [pvals[i] for i in self.FF.qmap]
            # All of a sudden I need the number of virtual sites.
            qatoms = zeros(self.nparticles)
            for i, jj in enumerate(self.FF.qid):
                for j in jj:
                    qatoms[j] = qvals[i]
            return qatoms

        # Obtain a derivative matrix the stupid way
        if AGrad:
            # dqPdqM = []
            # for i in range(np):
            #     print "Now working on parameter number", i
            #     dqPdqM.append(f12d3p(fdwrap(getqatoms,mvals,i), h = self.h)[0])
            # dqPdqM = mat(dqPdqM).T
            dqPdqM = mat([f12d3p(fdwrap(getqatoms,mvals,i), h = self.h)[0] for i in range(np)]).T
        xyzs = array(self.traj.xyzs)
        espqvals = array(self.espval)
        espxyz   = array(self.espxyz)

        ddVdqPdVS = {}
        # Second derivative of the inverse distance matrix with respect to the virtual site position
        dddVdqPdVS2 = {}
        if AGrad:
            for p in range(np):
                if 'VSITE' in self.FF.plist[p]:
                    ddVdqPdVS[p], dddVdqPdVS2[p] = f12d3p(fdwrap(self.build_invdist,mvals,p), h = self.h, f0 = self.invdists)
        X = 0
        D = 0
        G = zeros(np, dtype=float)
        H = zeros((np, np), dtype=float)
        for i in range(self.ns):
            P   = self.whamboltz_wts[i]
            Z  += P
            dVdqP   = mat(self.invdists[i])
            espqval = espqvals[i]
            espmval = dVdqP * col(getqatoms(mvals))
            desp    = flat(espmval) - espqval
            X      += P * dot(desp, desp) / self.nesp
            D      += P * (dot(espqval, espqval) / self.nesp - (sum(espqval) / self.nesp)**2)
            if AGrad:
                dVdqM   = (dVdqP * dqPdqM).T
                for p, vsd in ddVdqPdVS.items():
                    dVdqM[p,:] += flat(vsd[i] * col(getqatoms(mvals)))
                G      += flat(P * 2 * dVdqM * col(desp)) / self.nesp
                if AHess:
                    d2VdqM2 = zeros(dVdqM.shape, dtype=float)
                    for p, vsd in dddVdqPdVS2.items():
                        d2VdqM2[p,:] += flat(vsd[i] * col(getqatoms(mvals)))
                    H      += array(P * 2 * (dVdqM * dVdqM.T + d2VdqM2 * col(desp))) / self.nesp
        # Redundant but we keep it anyway
        D /= Z
        X /= Z
        X /= D
        G /= Z
        G /= D
        H /= Z
        H /= D
        if not in_fd():
            self.esp_err = sqrt(X)
        # Following is the restraint part
        # RESP hyperbola "strength" parameter; 0.0005 is weak, 0.001 is strong
        # RESP hyperbola "tightness" parameter; don't need to change this
        a = self.resp_a
        b = self.resp_b
        q = getqatoms(mvals)
        R   = a*sum((q**2 + b**2)**0.5 - b)
        dR  = a*q*(q**2 + b**2)**-0.5
        ddR = a*b**2*(q**2 + b**2)**-1.5
        self.respterm = R
        X += R
        if AGrad:
            G += flat(dqPdqM.T * col(dR))
            if AHess:
                H += diag(flat(dqPdqM.T * col(ddR)))
            
        Answer = {'X':X,'G':G,'H':H}
        return Answer

    def get_energy_force_(self, mvals, AGrad=False, AHess=False):
        if self.covariance:
            return self.get_energy_force_covariance_(mvals, AGrad, AHess)
        else:
            return self.get_energy_force_no_covariance_(mvals, AGrad, AHess)
        

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':zeros(self.FF.np, dtype=float), 'H':zeros((self.FF.np, self.FF.np), dtype=float)}
        tw = self.w_energy + self.w_force + self.w_resp
        if tw > 0.0:
            w_ef = (self.w_energy + self.w_force) / tw
            w_resp = self.w_resp / tw
        else:
            w_ef = 0.0
            w_resp = 0.0
        if self.energy or self.force:
            Answer_EF = self.get_energy_force_(mvals, AGrad, AHess)
            for i in Answer_EF:
                Answer[i] += w_ef * Answer_EF[i]
        if self.resp:
            Answer_ESP = self.get_resp_(mvals, AGrad, AHess)
            for i in Answer_ESP:
                Answer[i] += w_resp * Answer_ESP[i]
        if not any([self.energy, self.force, self.resp]):
            raise Exception("Ab initio fitting must have at least one of: Energy, Force, ESP")
        if not in_fd():
            self.objective = Answer['X']
        return Answer

def weighted_variance(SPiXi,WCiW,Z,L,R,NCP1):
    """ A more generalized version of build_objective which is
    callable for derivatives, but the covariance is not there anymore. """
    # These are the functions that we are building.
    X2        = 0.0
    # Divide by Z to normalize
    XiZ       = SPiXi/Z
    # Subtract out the average energy.
    XiZ[0] -= (L[0] * R[0])/Z/Z
    # Return the answer.
    X2      = dot(XiZ.flatten(),WCiW.flatten())
    return X2

def weighted_variance2(SPiXi,WCiW,Z,L,R,L2,R2,NCP1):
    """ A bit of a hack, since we have to subtract out two mean quantities to get Hessian elements. """
    # These are the functions that we are building.
    X2        = 0.0
    # Divide by Z to normalize
    XiZ       = SPiXi/Z
    # Subtract out the average energy.
    XiZ[0] -= (L[0] * R[0])/Z/Z
    XiZ[0] -= (L2[0] * R2[0])/Z/Z
    # Return the answer.
    X2      = dot(XiZ.flatten(),WCiW.flatten())
    return X2

def build_objective(SPiXi,WCiW,Z,Q0,M0,NCP1):

    """ This function builds an objective function (number) from the
    complicated polytensor and covariance matrices. """

    # These are the functions that we are building.
    X2    = 0.0
    # Divide by Z to normalize
    XiZ       = SPiXi/Z
    # Subtract out the zero-point energy gap
    XiZ[0,0] -= (M0[0]*M0[0] + Q0[0]*Q0[0] - 2*Q0[0]*M0[0])/Z/Z
    for i in range(1,NCP1):
        XiZ[0,i] -= (M0[i]*M0[0] + Q0[i]*Q0[0] - 2*Q0[i]*M0[0])/Z/Z
        XiZ[i,0] -= (M0[0]*M0[i] + Q0[0]*Q0[i] - 2*Q0[0]*M0[i])/Z/Z
    ### This is the objective function! LAAAAA ###
    X2      = dot(XiZ.flatten(),WCiW.flatten())
    return X2

