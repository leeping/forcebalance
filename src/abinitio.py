""" @package forcebalance.abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, warn_press_key
from numpy import append, array, cross, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt, hstack, sum, abs, vstack, max, arange
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import defaultdict, OrderedDict
import itertools
#from IPython import embed
#from _increment import AbInitio_Build

from forcebalance.output import getLogger
logger = getLogger(__name__)

class AbInitio(Target):

    """ Subclass of Target for fitting force fields to ab initio data.

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
    by subclasses like AbInitio_GMX."""
    
    def __init__(self,options,tgt_opts,forcefield):
        """
        Initialization; define a few core concepts.

        @todo Obtain the number of true atoms (or the particle -> atom mapping)
        from the force field.
        """

        ## Initialize the base class
        super(AbInitio,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        
        ## Number of snapshots
        self.set_option(tgt_opts,'shots','ns')
        ## Whether to use WHAM Boltzmann weights
        self.set_option(tgt_opts,'whamboltz','whamboltz')
        ## Whether to use the Sampling Correction
        self.set_option(tgt_opts,'sampcorr','sampcorr')
        ## Whether to match Absolute Energies (make sure you know what you're doing)
        self.set_option(tgt_opts,'absolute','absolute')
        ## Whether to use the Covariance Matrix
        self.set_option(tgt_opts,'covariance','covariance')
        ## Whether to use QM Boltzmann weights
        self.set_option(tgt_opts,'qmboltz','qmboltz')
        ## The temperature for QM Boltzmann weights
        self.set_option(tgt_opts,'qmboltztemp','qmboltztemp')
        ## Number of atoms that we are fitting
        self.set_option(tgt_opts,'fitatoms','fitatoms')
        ## Whether to fit Energies.
        self.set_option(tgt_opts,'energy','energy')
        ## Whether to fit Forces.
        self.set_option(tgt_opts,'force','force')
        ## Whether to fit Electrostatic Potential.
        self.set_option(tgt_opts,'resp','resp')
        self.set_option(tgt_opts,'resp_a','resp_a')
        self.set_option(tgt_opts,'resp_b','resp_b')
        ## Weights for the three components.
        self.set_option(tgt_opts,'w_energy','w_energy')
        self.set_option(tgt_opts,'w_force','w_force')
        self.set_option(tgt_opts,'force_map','force_map')
        self.set_option(tgt_opts,'w_netforce','w_netforce')
        self.set_option(tgt_opts,'w_torque','w_torque')
        self.set_option(tgt_opts,'w_resp','w_resp')
        ## Option for how much data to write to disk.
        self.set_option(tgt_opts,'writelevel','writelevel')
        ## Whether to do energy and force calculations for the whole trajectory, or to do
        ## one calculation per snapshot.
        self.set_option(tgt_opts,'all_at_once','all_at_once')
        ## OpenMM-only option - whether to run the energies and forces internally.
        self.set_option(tgt_opts,'run_internal','run_internal')
        ## Whether we have virtual sites (set at the global option level)
        self.set_option(options,'have_vsite','have_vsite')
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
        self.qfnm = os.path.join(self.tgtdir,"qdata.txt")
        ## The number of atoms in the QM calculation (Irrelevant if not fitting forces)
        self.qmatoms      = 0
        ## Qualitative Indicator: average energy error (in kJ/mol)
        self.e_err = 0.0
        self.e_err_pct = None
        ## Qualitative Indicator: average force error (fractional)
        self.f_err = 0.0
        self.f_err_pct = 0.0
        ## Qualitative Indicator: "relative RMS" for electrostatic potential
        self.esp_err = 0.0
        self.nf_err = 0.0
        self.nf_err_pct = 0.0
        self.tq_err_pct = 0.0
        ## Whether to compute net forces and torques, or not.
        self.use_nft       = self.w_netforce > 0 or self.w_torque > 0
        ## Read in the trajectory file
        if self.ns == -1:
            self.traj = Molecule(os.path.join(self.root,self.tgtdir,self.trajfnm))
            self.ns = len(self.traj)
        else:
            self.traj = Molecule(os.path.join(self.root,self.tgtdir,self.trajfnm))[:self.ns]
        ## The number of (atoms + drude particles + virtual sites)
        self.nparticles  = len(self.traj.elem)
        ## This is a default-dict containing a number of atom-wise lists, such as the
        ## residue number of each atom, the mass of each atom, and so on.
        self.AtomLists = defaultdict(list)
        ## Read in the topology
        self.read_topology()
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,tgt_opts)
        ## The below two options are related to whether we want to rebuild virtual site positions.
        ## Rebuild the distance matrix if virtual site positions have changed
        self.new_vsites = True
        ## Save the mvals from the last time we updated the vsites.
        self.save_vmvals = {}
        self.set_option(None, 'shots', val=self.ns)

    def read_topology(self):
        # Arthur: Document this.
        self.topology_flag = False

    def build_invdist(self, mvals):
        for i in range(self.FF.np):
            if 'VSITE' in self.FF.plist[i]:
                if i in self.save_vmvals and mvals[i] != self.save_vmvals[i]:
                    self.new_vsites = True
                    break
        if not self.new_vsites: return self.invdists
        if any(['VSITE' in i for i in self.FF.map.keys()]) or self.have_vsite:
            logger.info("\rGenerating virtual site positions.%s" % (" "*30))
            pvals = self.FF.make(mvals)
            self.generate_vsite_positions()
        # prepare the distance matrix for esp computations
        if len(self.espxyz) > 0:
            invdists = []
            logger.info("\rPreparing the distance matrix... it will have %i * %i * %i = %i elements" % (self.ns, self.nesp, self.nparticles, self.ns * self.nesp * self.nparticles))
            sn = 0
            for espset, xyz in zip(self.espxyz, self.traj.xyzs):
                logger.info("\rGenerating ESP distances for snapshot %i%s" % (sn, " "*50))
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

    def compute_netforce_torque(self, xyz, force, QM=False):
        # Convert an array of (3 * n_atoms) atomistic forces
        # to an array of (3 * (n_forces + n_torques)) net forces and torques.
        # This code is rather slow.  It requires the system to have a list
        # of masses and blocking numbers.
        if not self.topology_flag:
            raise Exception('Cannot do net forces and torques for class %s because read_topology is not implemented' % self.__class__.__name__)

        if self.force_map == 'molecule':
            Block = self.AtomLists['MoleculeNumber']
        elif self.force_map == 'residue':
            Block = self.AtomLists['ResidueNumber']
        elif self.force_map == 'chargegroup':
            Block = self.AtomLists['ChargeGroupNumber']
        else:
            raise Exception('Please choose a valid force_map keyword: molecule, residue, chargegroup')

        # Try to be intelligent here.  Before computing net forces and torques, first filter out all particles that are not atoms.
        if len(xyz) > self.fitatoms:
            xyz1 = array([xyz[i] for i in range(len(xyz)) if self.AtomLists['ParticleType'][i] == 'A'])
        else:
            xyz1 = xyz.copy()

        if len(Block) > self.fitatoms:
            Block = [Block[i] for i in range(len(Block)) if self.AtomLists['ParticleType'][i] == 'A']

        if len(self.AtomLists['Mass']) > self.fitatoms:
            Mass = array([self.AtomLists['Mass'][i] for i in range(len(xyz)) if self.AtomLists['ParticleType'][i] == 'A'])
        else:
            Mass = array(self.AtomLists['Mass'])

        NetForces = []
        Torques = []
        for b in sorted(set(Block)):
            AtomBlock = array([i for i in range(len(Block)) if Block[i] == b])
            CrdBlock = array(list(itertools.chain(*[range(3*i, 3*i+3) for i in AtomBlock])))
            com = sum(xyz1[AtomBlock]*outer(Mass[AtomBlock],ones(3)), axis=0) / sum(Mass[AtomBlock])
            frc = force[CrdBlock].reshape(-1,3)
            NetForce = sum(frc, axis=0)
            xyzb = xyz1[AtomBlock]
            Torque = zeros(3, dtype=float)
            for a in range(len(xyzb)):
                R = xyzb[a] - com
                F = frc[a]
                # I think the unit of torque is in nm x kJ / nm.
                Torque += cross(R, F) / 10
            NetForces += [i for i in NetForce]
            # Increment the torques only if we have more than one atom in our block.
            if len(xyzb) > 1:
                Torques += [i for i in Torque]
        netfrc_torque = array(NetForces + Torques)
        self.nnf = len(NetForces)/3
        self.ntq = len(Torques)/3
        return netfrc_torque

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
        if not self.absolute:
            self.eqm -= mean(self.eqm)
        else:
            logger.info("Fitting absolute energies.  Make sure you know what you are doing!\n")
        if len(self.fqm) > 0:
            self.fqm = array(self.fqm)
            self.fqm *= fqcgmx
            self.qmatoms = self.fqm.shape[1]/3
        else:
            logger.info("QM forces are not present, only fitting energies.\n")
            self.force = 0
            self.w_force = 0
        self.nesp = len(self.espval[0]) if len(self.espval) > 0 else 0
        # Here we may choose a subset of atoms to do the force matching.
        if self.force:
            if self.fitatoms == 0: 
                self.fitatoms = self.qmatoms
            elif self.fitatoms > self.qmatoms:
                warn_press_key("There are more fitting atoms than the total number of atoms in the QM calculation (something is probably wrong)")
            else:
                # Indicate to Gromacs that we're only fitting the first however-many atoms.
                logger.info("We're only fitting the first %i atoms\n" % self.fitatoms)
                #print "The quantum force matrix appears to contain more components (%i) than those being fit (%i)." % (fqmm.shape[1], 3*self.fitatoms)
                logger.info("Pruning the quantum force matrix...\n")
                self.fqm  = self.fqm[:, :3*self.fitatoms].copy()
        else:
            self.fitatoms = 0
            
        if len(self.emd0) > 0:
            self.emd0 = array(self.emd0)
            self.emd0 -= mean(self.emd0)
        if self.whamboltz == True:
            self.whamboltz_wts = array([float(i.strip()) for i in open(os.path.join(self.root,self.tgtdir,"wham-weights.txt")).readlines()])
            #   This is a constant pre-multiplier in front of every snapshot.
            bar = printcool("Using WHAM MM Boltzmann weights.", color=4)
            if os.path.exists(os.path.join(self.root,self.tgtdir,"wham-master.txt")):
                whaminfo = open(os.path.join(self.root,self.tgtdir,"wham-master.txt")).readlines()
                logger.info("From wham-master.txt, I can see that you're using %i generations\n" % len(whaminfo))
                logger.info("Relative weight of each generation:\n")
                shotcounter = 0
                for line in whaminfo:
                    sline = line.split()
                    genshots = int(sline[2])
                    weight = sum(self.whamboltz_wts[shotcounter:shotcounter+genshots])/sum(self.whamboltz_wts)
                    logger.info(" %s, %i snapshots, weight %.3e\n" % (sline[0], genshots, weight))
                    shotcounter += genshots
            else:
                logger.info("Oops... WHAM files don't exist?\n")
            logger.info(bar)
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
            logger.info("Quantum Boltzmann weights are ON, the formula is exp(-b(E_qm-E_mm)),")
            logger.info("distribution entropy is %.3f, equivalent to %.2f snapshots\n" % (qmboltzent, exp(qmboltzent)))
            logger.info("%.1f%% is mixed into the MM boltzmann weights.\n" % (self.qmboltz*100))
        else:
            self.qmboltz_wts = ones(self.ns)
        # At this point, self.fqm is a (number of snapshots) x (3 x number of atoms) array.
        # Now we can transform it into a (number of snapshots) x (3 x number of residues + 3 x number of residues) array.
        if self.use_nft:
            self.nftqm = []
            for i in range(len(self.fqm)):
                self.nftqm.append(self.compute_netforce_torque(self.traj.xyzs[i], self.fqm[i]))
            self.nftqm = array(self.nftqm)
            self.fref = hstack((self.fqm, self.nftqm))
        else:
            self.fref = self.fqm

    def prepare_temp_directory(self, options, tgt_opts):
        """ Prepare the temporary directory, by default does nothing """
        return
        
    def indicate(self):
        Headings = ["Physical Variable", "Difference\n(Calc-Ref)", "Denominator\n RMS (Ref)", " Percent \nDifference"]
        Data = OrderedDict([])
        if self.energy:
            Data['Energy (kJ/mol)'] = ["%8.4f" % self.e_err,
                                       "%8.4f" % self.e_ref,
                                       "%.4f%%" % (self.e_err_pct*100)]
        if self.force:
            Data['Gradient (kJ/mol/A)'] = ["%8.4f" % (self.f_err/10),
                                           "%8.4f" % (self.f_ref/10),
                                           "%.4f%%" % (self.f_err_pct*100)]
            if self.use_nft:
                Data['Net Force (kJ/mol/A)'] = ["%8.4f" % (self.nf_err/10),
                                                "%8.4f" % (self.nf_ref/10),
                                                "%.4f%%" % (self.nf_err_pct*100)]
                Data['Torque (kJ/mol/rad)'] = ["%8.4f" % self.tq_err,
                                               "%8.4f" % self.tq_ref,
                                               "%.4f%%" % (self.tq_err_pct*100)]
        self.printcool_table(data=Data, headings=Headings, color=0)

    def energy_force_transformer_all(self):
        M = self.energy_force_driver_all()
        if self.force:
            if self.use_nft:
                Nfts = []
                for i in range(len(M)):
                    Fm  = M[i][1:]
                    Nft = self.compute_netforce_torque(self.traj.xyzs[i], Fm)
                    Nfts.append(Nft)
                Nfts = array(Nfts)
                return hstack((M, Nfts))
            else:
                return M
        else:
            return M

    def energy_force_transformer(self,i):
        M = self.energy_force_driver(i)
        if self.force:
            if self.use_nft:
                Fm  = M[1:]
                Nft = self.compute_netforce_torque(self.traj.xyzs[i], Fm)
                return hstack((M, Nft))
            else:
                return M
        else:
            return M[0]

    def get_energy_force_(self, mvals, AGrad=False, AHess=False):
        """
        LPW 06-30-2013
        
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

        This version implements both the polytensor formulation and the standard formulation.

        1) Boltzmann weights and normalization: Each snapshot has its
        own Boltzmann weight, which may or may not be normalized.
        This subroutine does the normalization automatically.
        
        2) Subtracting out the mean energy gap: The zero-point energy
        difference between reference data and simulation is
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
        cv = self.covariance # Whether covariance matrix is turned on
        if cv and (AGrad or AHess):
            warn_press_key("Covariance is turned on, gradients requested but not implemented (will skip over grad.)")
        Answer = {}
        # Create the new force field!!
        pvals = self.FF.make(mvals)

        #======================================#
        #   Copied from the old ForTune code   #
        #======================================#
        NC   = 3*self.fitatoms
        NCP1 = 3*self.fitatoms+1
        if self.use_nft:
            NCP1 += 3*(self.nnf + self.ntq)
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
        # Later on we divide these quantities by Z to normalize,       #
        # and optionally Q0(X)Q0 is subtracted from QQ to get the      #
        # covariance.                                                  #
        #==============================================================#
        if (self.w_energy == 0.0 and self.w_force == 0.0 and self.w_netforce == 0.0 and self.w_torque == 0.0):
            AGrad = False
            AHess = False
        Z       = 0.0
        Y       = 0.0
        # The average force / net force per atom in kJ/mol/nm. (Added LPW 6/30)
        # Torques are in kj/mol/nm x nm.
        qF_M     = 0
        qN_M     = 0
        qT_M     = 0
        qF_Q     = 0
        qN_Q     = 0
        qT_Q     = 0
        dF_M     = 0
        dN_M     = 0
        dT_M     = 0
        dF_Q     = 0
        dN_Q     = 0
        dT_Q     = 0
        Q = zeros(NCP1,dtype=float)
        # Mean quantities
        M0_M    = zeros(NCP1,dtype=float)
        Q0_M    = zeros(NCP1,dtype=float)
        M0_Q    = zeros(NCP1,dtype=float)
        Q0_Q    = zeros(NCP1,dtype=float)
        if cv:
            QQ_M    = zeros((NCP1,NCP1),dtype=float)
            QQ_Q    = zeros((NCP1,NCP1),dtype=float)
            #==============================================================#
            # Objective function polytensors: This is formed in a loop     #
            # over snapshots by taking the outer product (Q-M)(X)(Q-M),    #
            # multiplying by the Boltzmann weight, and then summing.       #
            #==============================================================#
            SPiXi = zeros((NCP1,NCP1),dtype=float)
            SRiXi = zeros((NCP1,NCP1),dtype=float)
        else:
            # Derivatives
            M_p     = zeros((np,NCP1),dtype=float)
            M_pp    = zeros((np,NCP1),dtype=float)
            X0_M    = zeros(NCP1,dtype=float)
            QQ_M    = zeros(NCP1,dtype=float)
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
            M_all = zeros((ns,NCP1),dtype=float)
            if AGrad and self.all_at_once:
                dM_all = zeros((ns,np,NCP1),dtype=float)
                ddM_all = zeros((ns,np,NCP1),dtype=float)
        QBN = dot(self.qmboltz_wts[:ns],self.whamboltz_wts[:ns])
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        if self.all_at_once:
            logger.info("\rExecuting\r")
            M_all = self.energy_force_transformer_all()
            if not cv and (AGrad or AHess):
                def callM(mvals_):
                    logger.info("\r")
                    pvals = self.FF.make(mvals_)
                    return self.energy_force_transformer_all()
                for p in range(np):
                    dM_all[:,p,:], ddM_all[:,p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M_all)
        for i in range(ns):
            if i % 100 == 0:
                logger.info("\rIncrementing quantities for snapshot %i\r" % i)
            # Build Boltzmann weights and increment partition function.
            P   = self.whamboltz_wts[i]
            Z  += P
            R   = self.qmboltz_wts[i]*self.whamboltz_wts[i] / QBN
            Y  += R
            # Recall reference (QM) data
            Q[0] = self.eqm[i]
            if self.force:
                Q[1:] = self.fref[i,:].copy()
            # Increment the average quantities.
            if cv:
                QQ     = outer(Q,Q)
            else:
                QQ     = Q*Q
            # Call the simulation software to get the MM quantities.
            if self.all_at_once:
                M = M_all[i]
            else:
                if i % 100 == 0:
                    logger.info("Shot %i\r" % i)
                M = self.energy_force_transformer(i)
                M_all[i,:] = M.copy()
            if not cv:
                X     = M-Q
            # Increment the average values.
            if self.force:
                dfrcarray = mean(array([linalg.norm(M[1+3*j:1+3*j+3] - Q[1+3*j:1+3*j+3]) for j in range(self.fitatoms)]))
                qfrcarray = mean(array([linalg.norm(Q[1+3*j:1+3*j+3]) for j in range(self.fitatoms)]))
                dF_M    += P*dfrcarray
                dF_Q    += R*dfrcarray
                qF_M    += P*qfrcarray
                qF_Q    += R*qfrcarray
                if self.use_nft:
                    dnfrcarray = mean(array([linalg.norm(M[1+3*self.fitatoms+3*j:1+3*self.fitatoms+3*j+3] - Q[1+3*self.fitatoms+3*j:1+3*self.fitatoms+3*j+3]) for j in range(self.nnf)]))
                    qnfrcarray = mean(array([linalg.norm(Q[1+3*self.fitatoms+3*j:1+3*self.fitatoms+3*j+3]) for j in range(self.nnf)]))
                    dN_M    += P*dnfrcarray
                    dN_Q    += R*dnfrcarray
                    qN_M    += P*qnfrcarray
                    qN_Q    += R*qnfrcarray
                    dtrqarray = mean(array([linalg.norm(M[1+3*self.fitatoms+3*self.nnf+3*j:1+3*self.fitatoms+3*self.nnf+3*j+3] - Q[1+3*self.fitatoms+3*self.nnf+3*j:1+3*self.fitatoms+3*self.nnf+3*j+3]) for j in range(self.ntq)]))
                    qtrqarray = mean(array([linalg.norm(Q[1+3*self.fitatoms+3*self.nnf+3*j:1+3*self.fitatoms+3*self.nnf+3*j+3]) for j in range(self.ntq)]))
                    dT_M    += P*dtrqarray
                    dT_Q    += R*dtrqarray
                    qT_M    += P*qtrqarray
                    qT_Q    += R*qtrqarray
            # The [0] indicates that we are fitting the RMS force and not the RMSD
            # (without the covariance, subtracting a mean force doesn't make sense.)
            M0_M[0] += P*M[0]
            M0_Q[0] += R*M[0]
            Q0_M[0] += P*Q[0]
            Q0_Q[0] += R*Q[0]
            QQ_M += P*QQ
            QQ_Q += R*QQ
            if not cv:
                X0_M[0] += P*X[0]
                X0_Q[0] += R*X[0]
            # Increment the objective function.
            if cv:
                Xi  = outer(M,M) - 2*outer(Q,M) + outer(Q,Q)
            else:
                Xi     = X**2                   
            SPiXi += P * Xi
            SRiXi += R * Xi
            #==============================================================#
            #      STEP 2a: Increment gradients and mean quantities.       #
            #   This is only implemented for the case without covariance.  #
            #==============================================================#
            if not cv:
                for p in range(np):
                    if not AGrad: continue
                    if self.all_at_once:
                        M_p[p] = dM_all[i, p]
                        M_pp[p] = ddM_all[i, p]
                    else:
                        def callM(mvals_):
                            if i % 100 == 0:
                                logger.info("\r")
                            pvals = self.FF.make(mvals_)
                            return self.energy_force_transformer(i)
                        M_p[p],M_pp[p] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
                    # The [0] indicates that we are fitting the RMS force and not the RMSD
                    # (without the covariance, subtracting a mean force doesn't make sense.)
                    M0_M_p[p][0]  += P * M_p[p][0]
                    M0_Q_p[p][0]  += R * M_p[p][0]
                    #M0_M_pp[p][0] += P * M_pp[p][0]
                    #M0_Q_pp[p][0] += R * M_pp[p][0]
                    Xi_p        = 2 * X * M_p[p]
                    SPiXi_p[p] += P * Xi_p
                    SRiXi_p[p] += R * Xi_p
                    if not AHess: continue
                    if self.all_at_once:
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

        # Dump energies and forces to disk.
        M_all_print = M_all.copy()
        if not self.absolute:
            M_all_print[:,0] -= mean(M_all_print[:,0])
        if self.force:
            Q_all_print = hstack((col(self.eqm),self.fref))
        else:
            Q_all_print = col(self.eqm)
        if not self.absolute:
            Q_all_print[:,0] -= mean(Q_all_print[:,0])
        if self.writelevel > 1:
            savetxt('M.txt',M_all_print)
            savetxt('Q.txt',Q_all_print)
        EnergyComparison = hstack((col(Q_all_print[:,0]),col(M_all_print[:,0])))
        if self.writelevel > 0:
            savetxt('QM-vs-MM-energies.txt',EnergyComparison)
        if self.force:
            # Write .xyz files which can be viewed in vmd.
            try:
                # Only print forces on true atoms, and ignore virtual sites.
                TrueAtoms = array([i for i in range(self.traj.na) if self.AtomLists['ParticleType'][i] == 'A'])
            except:
                TrueAtoms = arange(self.traj.na)
            QMTraj = self.traj[:].atom_select(TrueAtoms)
            Mforce_obj = QMTraj[:]
            Qforce_obj = QMTraj[:]
            Mforce_print = array(M_all_print[:,1:3*self.fitatoms+1])
            Qforce_print = array(Q_all_print[:,1:3*self.fitatoms+1])
            Dforce_norm = array([linalg.norm(Mforce_print[i,:] - Qforce_print[i,:]) for i in range(ns)])
            MaxComp = max(abs(vstack((Mforce_print,Qforce_print)).flatten()))
            Mforce_print /= MaxComp
            Qforce_print /= MaxComp
            for i in range(ns):
                Mforce_obj.xyzs[i] = Mforce_print[i, :].reshape(-1,3)
                Qforce_obj.xyzs[i] = Qforce_print[i, :].reshape(-1,3)
            if self.fitatoms < self.qmatoms:
                Fpad = zeros((self.qmatoms - self.fitatoms, 3),dtype=float)
                Mforce_obj.xyzs[i] = vstack((Mforce_obj.xyzs[i], Fpad))
                Qforce_obj.xyzs[i] = vstack((Qforce_obj.xyzs[i], Fpad))
            if Mforce_obj.na != Mforce_obj.xyzs[0].shape[0]:
                warn_once('\x1b[91mThe printing of forces is not set up correctly.  Not printing forces.  Please report this issue.\x1b[0m')
            else:
                if self.writelevel > 1:
                    QMTraj.write('coords.xyz')
                    Mforce_obj.elem = ['H' for i in range(Mforce_obj.na)]
                    Mforce_obj.write('MMforce.xyz')
                    Qforce_obj.elem = ['H' for i in range(Qforce_obj.na)]
                    Qforce_obj.write('QMforce.xyz')
                #savetxt('Dforce_norm.dat', Dforce_norm)
                if self.writelevel > 0:
                    savetxt('Dforce_norm.dat', Dforce_norm)


        #==============================================================#
        #    STEP 3: Build the (co)variance matrix and invert it.      #
        # In the case of no covariance, this is just a diagonal matrix #
        # with the RMSD energy in [0,0] and the RMS gradient in [n, n] #
        #==============================================================#
        logger.info("Done with snapshots, building objective function now\r")
        if (self.w_energy > 0.0 or self.w_force > 0.0 or self.w_netforce > 0.0 or self.w_torque > 0.0):
            wtot    = self.w_energy + self.w_force + self.w_netforce + self.w_torque
            EWt     = self.w_energy / wtot
            FWt     = self.w_force / wtot
            NWt     = self.w_netforce / wtot
            TWt     = self.w_torque / wtot
        else:
            EWt = 0.0
            FWt = 0.0
            NWt = 0.0
            TWt = 0.0
        # Build the weight vector/matrix, so the force contribution is suppressed by 1/3N
        if cv:
            WM      = zeros((NCP1,NCP1),dtype=float)
            WM[0,0] = sqrt(EWt)
            start   = 1
            block   = 3*self.fitatoms
            end     = start + block
            for i in range(start, end):
                WM[i, i] = sqrt(FWt / block)
            if self.use_nft:
                start   = end
                block   = 3*self.nnf
                end     = start + block
                for i in range(start, end):
                    WM[i, i] = sqrt(NWt / block)
                start   = end
                block   = 3*self.ntq
                end     = start + block
                for i in range(start, end):
                    WM[i, i] = sqrt(TWt / block)
        else:
            WM      = zeros(NCP1,dtype=float)
            WM[0] = sqrt(EWt)
            if self.force:
                start   = 1
                block   = 3*self.fitatoms
                end     = start + block
                WM[start:end] = sqrt(FWt / block)
                if self.use_nft:
                    start   = end
                    block   = 3*self.nnf
                    end     = start + block
                    WM[start:end] = sqrt(NWt / block)
                    start   = end
                    block   = 3*self.ntq
                    end     = start + block
                    WM[start:end] = sqrt(TWt / block)
        if cv:
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
            Ci   = invert_svd(C)
            # Get rid of energy-force covariance
            for i in range(1,NCP1):
                Ci[0,i] = 0.0;
                Ci[i,0] = 0.0;
            WCiW = array(mat(WM) * mat(Ci) * mat(WM)) # Weighted covariance matrix.
        else:
            # Here we're just using the variance.
            QBP  = self.qmboltz
            MBP  = 1 - self.qmboltz
            C    = MBP*(QQ_M-Q0_M*Q0_M/Z)/Z + QBP*(QQ_Q-Q0_Q*Q0_Q/Y)/Y
            # Normalize the force components
            for i in range(1, len(C), 3):
                C[i:i+3] = mean(C[i:i+3])
            Ci    = 1. / C
            WCiW = WM * Ci * WM
        #==============================================================#
        # STEP 4: Build the objective function and its derivatives.    #
        # Note that building derivatives with covariance is not yet    #
        # implemented, but can use external warpper to differentiate   #
        #==============================================================#
        if cv:
            X2_M  = build_objective(SPiXi,WCiW,Z,Q0_M,M0_M,NCP1)
            X2_Q  = build_objective(SRiXi,WCiW,Y,Q0_Q,M0_Q,NCP1)
        else:
            X2_M  = weighted_variance(SPiXi,WCiW,Z,X0_M,X0_M,NCP1,subtract_mean = not self.absolute)
            X2_Q  = weighted_variance(SRiXi,WCiW,Y,X0_Q,X0_Q,NCP1,subtract_mean = not self.absolute)
            for p in range(np):
                if not AGrad: continue
                X2_M_p[p] = weighted_variance(SPiXi_p[p],WCiW,Z,2*X0_M,M0_M_p[p],NCP1,subtract_mean = not self.absolute)
                X2_Q_p[p] = weighted_variance(SRiXi_p[p],WCiW,Y,2*X0_Q,M0_Q_p[p],NCP1,subtract_mean = not self.absolute)
                if not AHess: continue
                X2_M_pq[p,p] = weighted_variance2(SPiXi_pq[p,p],WCiW,Z,2*M0_M_p[p],M0_M_p[p],2*X0_M,M0_M_pp[p],NCP1,subtract_mean = not self.absolute)
                X2_Q_pq[p,p] = weighted_variance2(SRiXi_pq[p,p],WCiW,Y,2*M0_Q_p[p],M0_Q_p[p],2*X0_Q,M0_Q_pp[p],NCP1,subtract_mean = not self.absolute)
                for q in range(p):
                    X2_M_pq[p,q] = weighted_variance(SPiXi_pq[p,q],WCiW,Z,2*M0_M_p[p],M0_M_p[q],NCP1,subtract_mean = not self.absolute)
                    X2_Q_pq[p,q] = weighted_variance(SRiXi_pq[p,q],WCiW,Y,2*M0_Q_p[p],M0_Q_p[q],NCP1,subtract_mean = not self.absolute)
                    # Get the other half of the Hessian matrix.
                    X2_M_pq[q,p] = X2_M_pq[p,q]
                    X2_Q_pq[q,p] = X2_Q_pq[p,q]
        #==============================================================#
        #            STEP 5: Build the return quantities.              #
        #==============================================================#
        # The objective function
        X2   = MBP * X2_M    + QBP * X2_Q
        if not cv:
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
        if not self.absolute:
            E0_M = (2*Q0_M[0]*M0_M[0] - Q0_M[0]*Q0_M[0] - M0_M[0]*M0_M[0])/Z/Z;
            E0_Q = (2*Q0_Q[0]*M0_Q[0] - Q0_Q[0]*Q0_Q[0] - M0_Q[0]*M0_Q[0])/Y/Y;
        else:
            E0_M = 0.0
            E0_Q = 0.0
        if cv:
            dE     = MBP * sqrt(SPiXi[0,0]/Z + E0_M) + QBP * sqrt(SRiXi[0,0]/Y + E0_Q)
        else:
            dE     = MBP * sqrt(SPiXi[0]/Z + E0_M) + QBP * sqrt(SRiXi[0]/Y + E0_Q)
        # Fractional energy error.
        dEfrac = MBP * sqrt((SPiXi[0]/Z + E0_M) / (QQ_M[0]/Z - Q0_M[0]**2/Z/Z)) + QBP * sqrt((SRiXi[0]/Y + E0_Q) / (QQ_Q[0]/Y - Q0_Q[0]**2/Y/Y))
        # Absolute and Fractional force error.
        if self.force:
            dF = MBP * dF_M / Z + QBP * dF_Q / Y
            qF = MBP * qF_M / Z + QBP * qF_Q / Y
            dFfrac = MBP * (dF_M/qF_M) + QBP * (dF_Q/qF_Q)
            # Old code for generating percentage force errors is less intuitive.
            # if cv:
            #     dFfrac = MBP * sqrt(mean(array([SPiXi[i,i]/QQ_M[i,i] for i in range(1,1+3*self.fitatoms) if abs(QQ_M[i,i]) > 1e-3 ]))) + \
            #         QBP * sqrt(mean(array([SRiXi[i,i]/QQ_Q[i,i] for i in range(1,1+3*self.fitatoms) if abs(QQ_Q[i,i]) > 1e-3])))
            # else:
            #     dFfrac = MBP * sqrt(mean(array([SPiXi[i]/QQ_M[i] for i in range(1,1+3*self.fitatoms) if abs(QQ_M[i]) > 1e-3]))) + \
            #         QBP * sqrt(mean(array([SRiXi[i]/QQ_Q[i] for i in range(1,1+3*self.fitatoms) if abs(QQ_Q[i]) > 1e-3])))
        if self.use_nft:
            dN = MBP * dN_M / Z + QBP * dN_Q / Y
            qN = MBP * qN_M / Z + QBP * qN_Q / Y
            dNfrac = MBP * dN_M / qN_M + QBP * dN_Q / qN_Q
            dT = MBP * dT_M / Z + QBP * dT_Q / Y
            qT = MBP * qT_M / Z + QBP * qT_Q / Y
            dTfrac = MBP * dT_M / qT_M + QBP * dT_Q / qT_Q
            # if cv:
            #     dNfrac = MBP * sqrt(mean(array([SPiXi[i,i]/QQ_M[i,i] for i in range(1+3*self.fitatoms, 1+3*(self.fitatoms+self.nnf))]))) + \
            #         QBP * sqrt(mean(array([SRiXi[i,i]/QQ_Q[i,i] for i in range(1+3*self.fitatoms, 1+3*(self.fitatoms+self.nnf))])))
            #     dTfrac = MBP * sqrt(mean(array([SPiXi[i,i]/QQ_M[i,i] for i in range(1+3*(self.fitatoms+self.nnf), 1+3*(self.fitatoms+self.nnf+self.ntq))]))) + \
            #         QBP * sqrt(mean(array([SRiXi[i,i]/QQ_Q[i,i] for i in range(1+3*(self.fitatoms+self.nnf), 1+3*(self.fitatoms+self.nnf+self.ntq))])))
            # else:
            #     dNfrac = MBP * sqrt(mean(array([SPiXi[i]/QQ_M[i] for i in range(1+3*self.fitatoms, 1+3*(self.fitatoms+self.nnf)) if abs(QQ_M[i]) > 1e-3]))) + \
            #         QBP * sqrt(mean(array([SRiXi[i]/QQ_Q[i] for i in range(1+3*self.fitatoms, 1+3*(self.fitatoms+self.nnf)) if abs(QQ_Q[i]) > 1e-3])))
            #     dTfrac = MBP * sqrt(mean(array([SPiXi[i]/QQ_M[i] for i in range(1+3*(self.fitatoms+self.nnf), 1+3*(self.fitatoms+self.nnf+self.ntq)) if abs(QQ_M[i]) > 1e-3]))) + \
            #         QBP * sqrt(mean(array([SRiXi[i]/QQ_Q[i] for i in range(1+3*(self.fitatoms+self.nnf), 1+3*(self.fitatoms+self.nnf+self.ntq)) if abs(QQ_Q[i]) > 1e-3])))
        # Save values to qualitative indicator if not inside finite difference code.
        if not in_fd():
            self.e_ref = MBP * sqrt(QQ_M[0]/Z - Q0_M[0]**2/Z/Z) + QBP * sqrt((QQ_Q[0]/Y - Q0_Q[0]**2/Y/Y))
            self.e_err = dE
            self.e_err_pct = dEfrac
            if self.force:
                self.f_ref = qF
                self.f_err = dF
                self.f_err_pct = dFfrac
            if self.use_nft:
                self.nf_ref = qN
                self.nf_err = dN
                self.nf_err_pct = dNfrac
                self.tq_ref = qT
                self.tq_err = dT
                self.tq_err_pct = dTfrac
            pvals = self.FF.make(mvals) # Write a force field that isn't perturbed by finite differences.
        if cv:
            Answer = {'X':X2, 'G':zeros(self.FF.np), 'H':zeros((self.FF.np,self.FF.np))}
        else:
            Answer = {'X':X2, 'G':G, 'H':H}
        return Answer

    def get_resp_(self, mvals, AGrad=False, AHess=False):
        """ Electrostatic potential fitting.  Implements the RESP objective function.  (In Python so obviously not optimized.) """
        if (self.w_resp == 0.0):
            AGrad = False
            AHess = False
        Answer = {}
        pvals = self.FF.make(mvals)

        # Build the distance matrix for ESP fitting.
        self.invdists = self.build_invdist(mvals)

        ns = self.ns
        np = self.FF.np
        Z = 0
        Y = 0
        def getqatoms(mvals_):
            """ This function takes the mathematical parameter values and returns the charges on the ATOMS (fancy mapping going on) """
            logger.info("\r")
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

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':zeros(self.FF.np, dtype=float), 'H':zeros((self.FF.np, self.FF.np), dtype=float)}
        tw = self.w_energy + self.w_force + self.w_netforce + self.w_torque + self.w_resp
        if tw > 0.0:
            w_ef = (self.w_energy + self.w_force + self.w_netforce + self.w_torque) / tw
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

def weighted_variance(SPiXi,WCiW,Z,L,R,NCP1,subtract_mean=True):
    """ A more generalized version of build_objective which is
    callable for derivatives, but the covariance is not there anymore. """
    # These are the functions that we are building.
    X2        = 0.0
    # Divide by Z to normalize
    XiZ       = SPiXi/Z
    # Subtract out the average energy.
    if subtract_mean:
        XiZ[0] -= (L[0] * R[0])/Z/Z
    # Return the answer.
    X2      = dot(XiZ.flatten(),WCiW.flatten())
    return X2

def weighted_variance2(SPiXi,WCiW,Z,L,R,L2,R2,NCP1,subtract_mean=True):
    """ A bit of a hack, since we have to subtract out two mean quantities to get Hessian elements. """
    # These are the functions that we are building.
    X2        = 0.0
    # Divide by Z to normalize
    XiZ       = SPiXi/Z
    # Subtract out the average energy.
    if subtract_mean:
        XiZ[0] -= (L[0] * R[0])/Z/Z
        XiZ[0] -= (L2[0] * R2[0])/Z/Z
    # Return the answer.
    X2      = dot(XiZ.flatten(),WCiW.flatten())
    return X2

def build_objective(SPiXi,WCiW,Z,Q0,M0,NCP1,subtract_mean=True):

    """ This function builds an objective function (number) from the
    complicated polytensor and covariance matrices. """

    # These are the functions that we are building.
    X2    = 0.0
    # Divide by Z to normalize
    XiZ       = SPiXi/Z
    if subtract_mean:
        # Subtract out the zero-point energy gap
        XiZ[0,0] -= (M0[0]*M0[0] + Q0[0]*Q0[0] - 2*Q0[0]*M0[0])/Z/Z
        for i in range(1,NCP1):
            XiZ[0,i] -= (M0[i]*M0[0] + Q0[i]*Q0[0] - 2*Q0[i]*M0[0])/Z/Z
            XiZ[i,0] -= (M0[0]*M0[i] + Q0[0]*Q0[i] - 2*Q0[0]*M0[i])/Z/Z
    ### This is the objective function! LAAAAA ###
    X2      = dot(XiZ.flatten(),WCiW.flatten())
    return X2

