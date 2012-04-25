""" @package forceenergymatch_gmxx2 Force and energy matching module.

@author Lee-Ping Wang
@date 12/2011
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros
from fitsim import FittingSimulation
from molecule import Molecule
from re import match
import subprocess
from subprocess import PIPE

trajfnm_default = {'GMX'    : "all.gro",
                   'GROMACS': "all.gro",
                   'TINKER' : "all.arc"
                   }

class ForceEnergyMatch(FittingSimulation):

    """ Subclass of FittingSimulation for force and energy matching.

    In force and energy matching, we introduce the following concepts:
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
    by subclasses like ForceEnergyMatch_GMXX2."""
    
    def __init__(self,options,sim_opts,forcefield):
        """Instantiation of the subclass.

        We begin by instantiating the superclass here and also
        defining a number of core concepts for energy / force
        matching.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """
        
        # Initialize the SuperClass!
        super(ForceEnergyMatch,self).__init__(options,sim_opts,forcefield)
        
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
        ## The proportion of energy vs. force.
        self.efweight      = sim_opts['efweight']
        ## Whether to do energy and force calculations for the whole trajectory, or to do
        ## one calculation per snapshot.
        self.all_at_once   = sim_opts['all_at_once']
        ## OpenMM-only option - whether to run the energies and forces internally.
        self.run_internal  = sim_opts['run_internal']
        
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
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.simdir,"qdata.txt")
        ## The number of true atoms 
        self.natoms      = 0
        ## Qualitative Indicator: average energy error (in kJ/mol)
        self.e_err = 0.0
        ## Qualitative Indicator: average force error (fractional)
        self.f_err = 0.0
        #======================================#
        #          UNDER DEVELOPMENT           #
        #======================================#
        ## Trajectory file name.  I wanted to put all default values in parser.py, but unfortunately
        ## I couldn't come up with a convenient way of making the default value depend on the
        ## value of another key.  So we are setting it here *sigh*
        self.trajfnm = sim_opts['trajfnm'] if sim_opts['trajfnm'] != None else trajfnm_default[self.software]
        ## Read in the trajectory file
        self.traj = Molecule(os.path.join(self.root,self.simdir,self.trajfnm))
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory()
        
    def read_reference_data(self):
        
        """ Read the reference data (for force and energy matching)
        from a file such as qdata.txt.

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
            if len(self.eqm) and len(self.emd0) and len(self.fqm) == self.ns:
                # Break out of the loop if we've already gotten "ns" number of energies and forces.
                break
        self.ns = len(self.eqm)
        # Turn everything into arrays, convert to kJ/mol, and subtract the mean energy from the energy arrays
        self.eqm = array(self.eqm)
        self.eqm *= eqcgmx
        self.eqm -= mean(self.eqm)
        self.fqm = array(self.fqm)
        self.fqm *= fqcgmx
        self.natoms = self.fqm.shape[1]/3
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
            
    def indicate(self):
        print "Sim: %-15s E_err(kJ/mol)= %10.4f F_err(%%)= %10.4f" % (self.name, self.e_err, self.f_err*100)

    def get(self, mvals, AGrad=False, AHess=False, tempdir=None):
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
        @param[in] AGrad Switch to turn on analytic gradient, useless here
        @param[in] AHess Switch to turn on analytic Hessian, useless here
        @param[in] tempdir Temporary directory for running computation
        @return Answer Contribution to the objective function
        """
        if tempdir == None:
            tempdir = self.tempdir
        abstempdir = os.path.join(self.root,self.tempdir)
        Answer = {}
        cwd = os.getcwd()
        # Create the new force field!!
        pvals = self.FF.make(tempdir,mvals,self.usepvals)
        # Go into the temporary directory
        os.chdir(os.path.join(self.root,tempdir))

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
        QBN = dot(self.qmboltz_wts[:self.ns],self.whamboltz_wts[:self.ns])
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        if self.all_at_once:
            print "Computing forces\r",
            M_all = self.energy_force_driver_all()
        for i in range(self.ns):
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
        # The "two" is of no importance, it has been kept historically.
        EFW     = 2 *self.efweight
        CEFW    = 2 - EFW
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
        Efrac = MBP * sqrt(SPiXi[0,0]/QQ_M[0,0]) + QBP * sqrt(SRiXi[0,0]/QQ_Q[0,0])
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
        self.e_err = E
        self.f_err = F
        Answer = {'X':BC, 'G':zeros(self.FF.np), 'H':zeros((self.FF.np,self.FF.np))}
        os.chdir(cwd)
        return Answer

def build_objective(SPiXi,WCiW,Z,Q0,M0,NCP1):
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

    def prepare_temp_directory(self,tempdir=None):
        """ Prepare the temporary directory for running the external program.

        This method creates the temporary directory, links in the
        necessary files for running (except for the force field), and
        writes the coordinate file for the snapshots we've chosen.
        
        """
        
        if tempdir == None:
            tempdir = self.tempdir
        # Create the temporary directory
        abstempdir = os.path.join(self.root,self.tempdir)
        os.makedirs(abstempdir)
        
        if self.software in ['GMX','GROMACS']:
            # Link the necessary programs into the temporary directory
            os.symlink(os.path.join(self.gmxrunpath,"mdrun"+self.gmxsuffix),os.path.join(abstempdir,"mdrun"))
            os.symlink(os.path.join(self.gmxrunpath,"grompp"+self.gmxsuffix),os.path.join(abstempdir,"grompp"))
            os.symlink(os.path.join(self.gmxtoolpath,"g_energy"+self.gmxsuffix),os.path.join(abstempdir,"g_energy"))
            os.symlink(os.path.join(self.gmxtoolpath,"g_traj"+self.gmxsuffix),os.path.join(abstempdir,"g_traj"))
            # Link the run files
            os.symlink(os.path.join(self.root,self.simdir,"shot.mdp"),os.path.join(abstempdir,"shot.mdp"))
            # Write the trajectory to the temp-directory
            self.traj.write(os.path.join(abstempdir,"all.gro"))
            os.symlink(os.path.join(self.root,self.simdir,"topol.top"),os.path.join(abstempdir,"topol.top"))
            # Print out the first conformation in all.gro to use as conf.gro
            self.traj.write(os.path.join(abstempdir,"conf.gro"),subset=[0])
        elif self.software == ['TINKER']:
            ## LPW STILL IN DEVELOPMENT!
            pass
