""" @package forcebalance.abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
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
            self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords))
            self.ns = len(self.mol)
        else:
            self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords))[:self.ns]
        ## The number of (atoms + drude particles + virtual sites)
        self.nparticles  = len(self.mol.elem)
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
            self.mol.xyzs = self.engine.generate_vsite_positions()
        # prepare the distance matrix for esp computations
        if len(self.espxyz) > 0:
            invdists = []
            logger.info("\rPreparing the distance matrix... it will have %i * %i * %i = %i elements" % (self.ns, self.nesp, self.nparticles, self.ns * self.nesp * self.nparticles))
            sn = 0
            for espset, xyz in zip(self.espxyz, self.mol.xyzs):
                logger.info("\rGenerating ESP distances for snapshot %i%s" % (sn, " "*50))
                esparr = np.array(espset).reshape(-1,3)
                # Create a matrix with Nesp rows and Natoms columns.
                DistMat = np.array([[np.linalg.norm(i - j) for j in xyz] for i in esparr])
                invdists.append(1. / (DistMat / bohrang))
                sn += 1
        for i in range(self.FF.np):
            if 'VSITE' in self.FF.plist[i]:
                self.save_vmvals[i] = mvals[i]
        self.new_vsites = False
        return np.array(invdists)

    def compute_netforce_torque(self, xyz, force, QM=False):
        # Convert an array of (3 * n_atoms) atomistic forces
        # to an array of (3 * (n_forces + n_torques)) net forces and torques.
        # This code is rather slow.  It requires the system to have a list
        # of masses and blocking numbers.
        if not self.topology_flag:
            raise Exception('Cannot do net forces and torques for class %s because read_topology is not implemented' % self.__class__.__name__)
        
        if self.force_map == 'molecule' and 'MoleculeNumber' in self.AtomLists:
            Block = self.AtomLists['MoleculeNumber']
        elif self.force_map == 'residue' and 'ResidueNumber' in self.AtomLists:
            Block = self.AtomLists['ResidueNumber']
        elif self.force_map == 'chargegroup' and 'ChargeGroupNumber' in self.AtomLists:
            Block = self.AtomLists['ChargeGroupNumber']
        else:
            raise Exception('Please choose a valid force_map keyword: %s' % ', '.join(self.AtomLists.keys()))

        # Try to be intelligent here.  Before computing net forces and torques, first filter out all particles that are not atoms.
        if len(xyz) > self.fitatoms:
            xyz1 = np.array([xyz[i] for i in range(len(xyz)) if self.AtomLists['ParticleType'][i] == 'A'])
        else:
            xyz1 = xyz.copy()

        if len(Block) > self.fitatoms:
            Block = [Block[i] for i in range(len(Block)) if self.AtomLists['ParticleType'][i] == 'A']

        if len(self.AtomLists['Mass']) > self.fitatoms:
            Mass = np.array([self.AtomLists['Mass'][i] for i in range(len(xyz)) if self.AtomLists['ParticleType'][i] == 'A'])
        else:
            Mass = np.array(self.AtomLists['Mass'])

        NetForces = []
        Torques = []
        for b in sorted(set(Block)):
            AtomBlock = np.array([i for i in range(len(Block)) if Block[i] == b])
            CrdBlock = np.array(list(itertools.chain(*[range(3*i, 3*i+3) for i in AtomBlock])))
            com = np.sum(xyz1[AtomBlock]*np.outer(Mass[AtomBlock],np.ones(3)), axis=0) / np.sum(Mass[AtomBlock])
            frc = force[CrdBlock].reshape(-1,3)
            NetForce = np.sum(frc, axis=0)
            xyzb = xyz1[AtomBlock]
            Torque = np.zeros(3)
            for a in range(len(xyzb)):
                R = xyzb[a] - com
                F = frc[a]
                # I think the unit of torque is in nm x kJ / nm.
                Torque += np.cross(R, F) / 10
            NetForces += [i for i in NetForce]
            # Increment the torques only if we have more than one atom in our block.
            if len(xyzb) > 1:
                Torques += [i for i in Torque]
        netfrc_torque = np.array(NetForces + Torques)
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
        self.eqm = np.array(self.eqm)
        self.eqm *= eqcgmx
        if not self.absolute:
            self.eqm -= np.mean(self.eqm)
        else:
            logger.info("Fitting absolute energies.  Make sure you know what you are doing!\n")
        if len(self.fqm) > 0:
            self.fqm = np.array(self.fqm)
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
            self.emd0 = np.array(self.emd0)
            self.emd0 -= np.mean(self.emd0)
        if self.whamboltz == True:
            self.whamboltz_wts = np.array([float(i.strip()) for i in open(os.path.join(self.root,self.tgtdir,"wham-weights.txt")).readlines()])
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
                    weight = np.sum(self.whamboltz_wts[shotcounter:shotcounter+genshots])/np.sum(self.whamboltz_wts)
                    logger.info(" %s, %i snapshots, weight %.3e\n" % (sline[0], genshots, weight))
                    shotcounter += genshots
            else:
                logger.info("Oops... WHAM files don't exist?\n")
            logger.info(bar)
        else:
            self.whamboltz_wts = np.ones(self.ns)
        if self.qmboltz > 0:
            # LPW I haven't revised this section yet
            # Do I need to?
            logboltz = -(self.eqm - np.mean(self.eqm) - self.emd0 + np.mean(self.emd0)) / kb / self.qmboltztemp
            logboltz -= np.max(logboltz) # Normalizes boltzmann weights
            self.qmboltz_wts = np.exp(logboltz)
            self.qmboltz_wts /= np.sum(self.qmboltz_wts)
            # where simply gathers the nonzero values otherwise we get lots of NaNs
            qbwhere = self.qmboltz_wts[np.where(self.qmboltz_wts)[0]]
            # Compute ze InfoContent!
            qmboltzent = -np.sum(qbwhere*np.log(qbwhere))
            logger.info("Quantum Boltzmann weights are ON, the formula is exp(-b(E_qm-E_mm)),")
            logger.info("distribution entropy is %.3f, equivalent to %.2f snapshots\n" % (qmboltzent, np.exp(qmboltzent)))
            logger.info("%.1f%% is mixed into the MM boltzmann weights.\n" % (self.qmboltz*100))
        else:
            self.qmboltz_wts = np.ones(self.ns)
        # At this point, self.fqm is a (number of snapshots) x (3 x number of atoms) array.
        # Now we can transform it into a (number of snapshots) x (3 x number of residues + 3 x number of residues) array.
        if self.use_nft:
            self.nftqm = []
            for i in range(len(self.fqm)):
                self.nftqm.append(self.compute_netforce_torque(self.mol.xyzs[i], self.fqm[i]))
            self.nftqm = np.array(self.nftqm)
            self.fref = np.hstack((self.fqm, self.nftqm))
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

    def energy_all(self):
        if hasattr(self, 'engine'):
            return self.engine.energy().reshape(-1,1)
        else:
            raise NotImplementedError("Target must contain an engine object")

    def energy_force_all(self):
        if hasattr(self, 'engine'):
            return self.engine.energy_force()
        else:
            raise NotImplementedError("Target must contain an engine object")
        
    def energy_force_transform(self):
        if self.force:
            M = self.energy_force_all()
            M = M[:, :3*self.fitatoms+1]
            if self.use_nft:
                Nfts = []
                for i in range(len(M)):
                    Fm  = M[i][1:]
                    Nft = self.compute_netforce_torque(self.mol.xyzs[i], Fm)
                    Nfts.append(Nft)
                Nfts = np.array(Nfts)
                return np.hstack((M, Nfts))
            else:
                return M
        else:
            return self.energy_all()

    def energy_one(self, i):
        if hasattr(self, 'engine'):
            return self.engine.energy_one(i)
        else:
            raise NotImplementedError("Target must contain an engine object")

    def energy_force_one(self, i):
        if hasattr(self, 'engine'):
            return self.engine.energy_force_one(i)
        else:
            raise NotImplementedError("Target must contain an engine object")

    def energy_force_transform_one(self,i):
        if self.force:
            M = self.energy_force_one(i)
            M = M[:3*self.fitatoms+1]
            if self.use_nft:
                Fm  = M[1:]
                Nft = self.compute_netforce_torque(self.mol.xyzs[i], Fm)
                return np.hstack((M, Nft))
            else:
                return M
        else:
            return self.energy_one()

    def get_energy_force(self, mvals, AGrad=False, AHess=False):
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
        fat  = self.fitatoms
        NC   = 3*fat
        NCP1 = 3*fat+1
        if self.use_nft:
            NCP1 += 3*(self.nnf + self.ntq)
        NP   = self.FF.np
        NS   = self.ns
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
        Q = np.zeros(NCP1)
        # Mean quantities
        M0_M    = np.zeros(NCP1)
        Q0_M    = np.zeros(NCP1)
        M0_Q    = np.zeros(NCP1)
        Q0_Q    = np.zeros(NCP1)
        if cv:
            QQ_M    = np.zeros((NCP1,NCP1))
            QQ_Q    = np.zeros((NCP1,NCP1))
            #==============================================================#
            # Objective function polytensors: This is formed in a loop     #
            # over snapshots by taking the outer product (Q-M)(X)(Q-M),    #
            # multiplying by the Boltzmann weight, and then summing.       #
            #==============================================================#
            SPiXi = np.zeros((NCP1,NCP1))
            SRiXi = np.zeros((NCP1,NCP1))
        else:
            # Derivatives
            M_p     = np.zeros((NP,NCP1))
            M_pp    = np.zeros((NP,NCP1))
            X0_M    = np.zeros(NCP1)
            QQ_M    = np.zeros(NCP1)
            X0_Q    = np.zeros(NCP1)
            Q0_Q    = np.zeros(NCP1)
            QQ_Q    = np.zeros(NCP1)
            # Means of gradients
            M0_M_p  = np.zeros((NP,NCP1))
            M0_Q_p  = np.zeros((NP,NCP1))
            M0_M_pp = np.zeros((NP,NCP1))
            M0_Q_pp = np.zeros((NP,NCP1))
            # Objective functions
            SPiXi = np.zeros(NCP1)
            SRiXi = np.zeros(NCP1)
            if AGrad:
                SPiXi_p = np.zeros((NP,NCP1))
                SRiXi_p = np.zeros((NP,NCP1))
                X2_M_p = np.zeros(NP)
                X2_Q_p = np.zeros(NP)
            if AHess:
                SPiXi_pq = np.zeros((NP,NP,NCP1))
                SRiXi_pq = np.zeros((NP,NP,NCP1))
                X2_M_pq = np.zeros((NP,NP))
                X2_Q_pq = np.zeros((NP,NP))
            M_all = np.zeros((NS,NCP1))
            if AGrad and self.all_at_once:
                dM_all = np.zeros((NS,NP,NCP1))
                ddM_all = np.zeros((NS,NP,NCP1))
        QBN = np.dot(self.qmboltz_wts[:NS],self.whamboltz_wts[:NS])
        #==============================================================#
        #             STEP 2: Loop through the snapshots.              #
        #==============================================================#
        if self.all_at_once:
            logger.info("\rExecuting\r")
            M_all = self.energy_force_transform()
            if not cv and (AGrad or AHess):
                def callM(mvals_):
                    logger.info("\r")
                    pvals = self.FF.make(mvals_)
                    return self.energy_force_transform()
                for p in range(NP):
                    dM_all[:,p,:], ddM_all[:,p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M_all)
        for i in range(NS):
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
                QQ     = np.outer(Q,Q)
            else:
                QQ     = Q*Q
            # Call the simulation software to get the MM quantities.
            if self.all_at_once:
                M = M_all[i]
            else:
                if i % 100 == 0:
                    logger.info("Shot %i\r" % i)
                M = self.energy_force_transform_one(i)
                M_all[i,:] = M.copy()
            if not cv:
                X     = M-Q
            # Increment the average values.
            if self.force:
                dfrcarray = np.mean(np.array([np.linalg.norm(M[1+3*j:1+3*j+3] - Q[1+3*j:1+3*j+3]) for j in range(fat)]))
                qfrcarray = np.mean(np.array([np.linalg.norm(Q[1+3*j:1+3*j+3]) for j in range(fat)]))
                dF_M    += P*dfrcarray
                dF_Q    += R*dfrcarray
                qF_M    += P*qfrcarray
                qF_Q    += R*qfrcarray
                if self.use_nft:
                    dnfrcarray = np.mean(np.array([np.linalg.norm(M[1+3*fat+3*j:1+3*fat+3*j+3] - Q[1+3*fat+3*j:1+3*fat+3*j+3]) for j in range(self.nnf)]))
                    qnfrcarray = np.mean(np.array([np.linalg.norm(Q[1+3*fat+3*j:1+3*fat+3*j+3]) for j in range(self.nnf)]))
                    dN_M    += P*dnfrcarray
                    dN_Q    += R*dnfrcarray
                    qN_M    += P*qnfrcarray
                    qN_Q    += R*qnfrcarray
                    dtrqarray = np.mean(np.array([np.linalg.norm(M[1+3*fat+3*self.nnf+3*j:1+3*fat+3*self.nnf+3*j+3] - Q[1+3*fat+3*self.nnf+3*j:1+3*fat+3*self.nnf+3*j+3]) for j in range(self.ntq)]))
                    qtrqarray = np.mean(np.array([np.linalg.norm(Q[1+3*fat+3*self.nnf+3*j:1+3*fat+3*self.nnf+3*j+3]) for j in range(self.ntq)]))
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
                Xi  = np.outer(M,M) - 2*np.outer(Q,M) + np.outer(Q,Q)
            else:
                Xi     = X**2                   
            SPiXi += P * Xi
            SRiXi += R * Xi
            #==============================================================#
            #      STEP 2a: Increment gradients and mean quantities.       #
            #   This is only implemented for the case without covariance.  #
            #==============================================================#
            if not cv:
                for p in range(NP):
                    if not AGrad: continue
                    if self.all_at_once:
                        M_p[p] = dM_all[i, p]
                        M_pp[p] = ddM_all[i, p]
                    else:
                        def callM(mvals_):
                            if i % 100 == 0:
                                logger.info("\r")
                            pvals = self.FF.make(mvals_)
                            return self.energy_force_transform_one(i)
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
            M_all_print[:,0] -= np.mean(M_all_print[:,0])
        if self.force:
            Q_all_print = np.hstack((col(self.eqm),self.fref))
        else:
            Q_all_print = col(self.eqm)
        if not self.absolute:
            Q_all_print[:,0] -= np.mean(Q_all_print[:,0])
        if self.writelevel > 1:
            np.savetxt('M.txt',M_all_print)
            np.savetxt('Q.txt',Q_all_print)
        EnergyComparison = np.hstack((col(Q_all_print[:,0]),col(M_all_print[:,0])))
        if self.writelevel > 0:
            np.savetxt('QM-vs-MM-energies.txt',EnergyComparison)
        if self.force:
            # Write .xyz files which can be viewed in vmd.
            try:
                # Only print forces on true atoms, and ignore virtual sites.
                TrueAtoms = np.array([i for i in range(self.mol.na) if self.AtomLists['ParticleType'][i] == 'A'])
            except:
                TrueAtoms = np.arange(self.mol.na)
            QMTraj = self.mol[:].atom_select(TrueAtoms)
            Mforce_obj = QMTraj[:]
            Qforce_obj = QMTraj[:]
            Mforce_print = np.array(M_all_print[:,1:3*fat+1])
            Qforce_print = np.array(Q_all_print[:,1:3*fat+1])
            Dforce_norm = np.array([np.linalg.norm(Mforce_print[i,:] - Qforce_print[i,:]) for i in range(NS)])
            MaxComp = np.max(np.abs(np.vstack((Mforce_print,Qforce_print)).flatten()))
            Mforce_print /= MaxComp
            Qforce_print /= MaxComp
            for i in range(NS):
                Mforce_obj.xyzs[i] = Mforce_print[i, :].reshape(-1,3)
                Qforce_obj.xyzs[i] = Qforce_print[i, :].reshape(-1,3)
            if fat < self.qmatoms:
                Fpad = np.zeros((self.qmatoms - fat, 3))
                Mforce_obj.xyzs[i] = np.vstack((Mforce_obj.xyzs[i], Fpad))
                Qforce_obj.xyzs[i] = np.vstack((Qforce_obj.xyzs[i], Fpad))
            if Mforce_obj.na != Mforce_obj.xyzs[0].shape[0]:
                warn_once('\x1b[91mThe printing of forces is not set up correctly.  Not printing forces.  Please report this issue.\x1b[0m')
            else:
                if self.writelevel > 1:
                    QMTraj.write('coords.xyz')
                    Mforce_obj.elem = ['H' for i in range(Mforce_obj.na)]
                    Mforce_obj.write('MMforce.xyz')
                    Qforce_obj.elem = ['H' for i in range(Qforce_obj.na)]
                    Qforce_obj.write('QMforce.xyz')
                #np.savetxt('Dforce_norm.dat', Dforce_norm)
                if self.writelevel > 0:
                    np.savetxt('Dforce_norm.dat', Dforce_norm)


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
            WM      = np.zeros((NCP1,NCP1))
            WM[0,0] = np.sqrt(EWt)
            start   = 1
            block   = 3*fat
            end     = start + block
            for i in range(start, end):
                WM[i, i] = np.sqrt(FWt / block)
            if self.use_nft:
                start   = end
                block   = 3*self.nnf
                end     = start + block
                for i in range(start, end):
                    WM[i, i] = np.sqrt(NWt / block)
                start   = end
                block   = 3*self.ntq
                end     = start + block
                for i in range(start, end):
                    WM[i, i] = np.sqrt(TWt / block)
        else:
            WM      = np.zeros(NCP1)
            WM[0] = np.sqrt(EWt)
            if self.force:
                start   = 1
                block   = 3*fat
                end     = start + block
                WM[start:end] = np.sqrt(FWt / block)
                if self.use_nft:
                    start   = end
                    block   = 3*self.nnf
                    end     = start + block
                    WM[start:end] = np.sqrt(NWt / block)
                    start   = end
                    block   = 3*self.ntq
                    end     = start + block
                    WM[start:end] = np.sqrt(TWt / block)
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
            C    = MBP*(QQ_M-np.outer(Q0_M,Q0_M)/Z)/Z + QBP*(QQ_Q-np.outer(Q0_Q,Q0_Q)/Y)/Y
            Ci   = invert_svd(C)
            # Get rid of energy-force covariance
            for i in range(1,NCP1):
                Ci[0,i] = 0.0;
                Ci[i,0] = 0.0;
            WCiW = np.array(np.mat(WM) * np.mat(Ci) * np.mat(WM)) # Weighted covariance matrix.
        else:
            # Here we're just using the variance.
            QBP  = self.qmboltz
            MBP  = 1 - self.qmboltz
            C    = MBP*(QQ_M-Q0_M*Q0_M/Z)/Z + QBP*(QQ_Q-Q0_Q*Q0_Q/Y)/Y
            # Normalize the force components
            for i in range(1, len(C), 3):
                C[i:i+3] = np.mean(C[i:i+3])
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
            for p in range(NP):
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
            G = np.zeros(NP)
            H = np.zeros((NP,NP))
            for p in range(NP):
                if not AGrad: continue
                G[p] = MBP * X2_M_p[p] + QBP * X2_Q_p[p]
                if not AHess: continue
                for q in range(NP):
                    H[p,q] = MBP * X2_M_pq[p,q] + QBP * X2_Q_pq[p,q]
        # Energy error in kJ/mol
        if not self.absolute:
            E0_M = (2*Q0_M[0]*M0_M[0] - Q0_M[0]*Q0_M[0] - M0_M[0]*M0_M[0])/Z/Z;
            E0_Q = (2*Q0_Q[0]*M0_Q[0] - Q0_Q[0]*Q0_Q[0] - M0_Q[0]*M0_Q[0])/Y/Y;
        else:
            E0_M = 0.0
            E0_Q = 0.0
        if cv:
            dE     = MBP * np.sqrt(SPiXi[0,0]/Z + E0_M) + QBP * np.sqrt(SRiXi[0,0]/Y + E0_Q)
        else:
            dE     = MBP * np.sqrt(SPiXi[0]/Z + E0_M) + QBP * np.sqrt(SRiXi[0]/Y + E0_Q)
        # Fractional energy error.
        dEfrac = MBP * np.sqrt((SPiXi[0]/Z + E0_M) / (QQ_M[0]/Z - Q0_M[0]**2/Z/Z)) + QBP * np.sqrt((SRiXi[0]/Y + E0_Q) / (QQ_Q[0]/Y - Q0_Q[0]**2/Y/Y))
        # Absolute and Fractional force error.
        if self.force:
            dF = MBP * dF_M / Z + QBP * dF_Q / Y
            qF = MBP * qF_M / Z + QBP * qF_Q / Y
            dFfrac = MBP * (dF_M/qF_M) + QBP * (dF_Q/qF_Q)
        if self.use_nft:
            dN = MBP * dN_M / Z + QBP * dN_Q / Y
            qN = MBP * qN_M / Z + QBP * qN_Q / Y
            dNfrac = MBP * dN_M / qN_M + QBP * dN_Q / qN_Q
            dT = MBP * dT_M / Z + QBP * dT_Q / Y
            qT = MBP * qT_M / Z + QBP * qT_Q / Y
            dTfrac = MBP * dT_M / qT_M + QBP * dT_Q / qT_Q
        # Save values to qualitative indicator if not inside finite difference code.
        if not in_fd():
            self.e_ref = MBP * np.sqrt(QQ_M[0]/Z - Q0_M[0]**2/Z/Z) + QBP * np.sqrt((QQ_Q[0]/Y - Q0_Q[0]**2/Y/Y))
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
            Answer = {'X':X2, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np,self.FF.np))}
        else:
            Answer = {'X':X2, 'G':G, 'H':H}
        return Answer

    def get_resp(self, mvals, AGrad=False, AHess=False):
        """ Electrostatic potential fitting.  Implements the RESP objective function.  (In Python so obviously not optimized.) """
        if (self.w_resp == 0.0):
            AGrad = False
            AHess = False
        Answer = {}
        pvals = self.FF.make(mvals)

        # Build the distance matrix for ESP fitting.
        self.invdists = self.build_invdist(mvals)

        NS = self.ns
        NP = self.FF.np
        Z = 0
        Y = 0
        def getqatoms(mvals_):
            """ This function takes the mathematical parameter values and returns the charges on the ATOMS (fancy mapping going on) """
            logger.info("\r")
            # Need to update the positions of atoms, if there are virtual sites.
            pvals = self.FF.create_pvals(mvals_)
            qvals = [pvals[i] for i in self.FF.qmap]
            # All of a sudden I need the number of virtual sites.
            qatoms = np.zeros(self.nparticles)
            for i, jj in enumerate(self.FF.qid):
                for j in jj:
                    qatoms[j] = qvals[i]
            return qatoms

        # Obtain a derivative matrix the stupid way
        if AGrad:
            # dqPdqM = []
            # for i in range(NP):
            #     print "Now working on parameter number", i
            #     dqPdqM.append(f12d3p(fdwrap(getqatoms,mvals,i), h = self.h)[0])
            # dqPdqM = mat(dqPdqM).T
            dqPdqM = np.mat([f12d3p(fdwrap(getqatoms,mvals,i), h = self.h)[0] for i in range(NP)]).T
        xyzs = np.array(self.mol.xyzs)
        espqvals = np.array(self.espval)
        espxyz   = np.array(self.espxyz)

        ddVdqPdVS = {}
        # Second derivative of the inverse distance matrix with respect to the virtual site position
        dddVdqPdVS2 = {}
        if AGrad:
            for p in range(NP):
                if 'VSITE' in self.FF.plist[p]:
                    ddVdqPdVS[p], dddVdqPdVS2[p] = f12d3p(fdwrap(self.build_invdist,mvals,p), h = self.h, f0 = self.invdists)
        X = 0
        D = 0
        G = np.zeros(NP)
        H = np.zeros((NP, NP))
        for i in range(self.ns):
            P   = self.whamboltz_wts[i]
            Z  += P
            dVdqP   = np.mat(self.invdists[i])
            espqval = espqvals[i]
            espmval = dVdqP * col(getqatoms(mvals))
            desp    = flat(espmval) - espqval
            X      += P * np.dot(desp, desp) / self.nesp
            D      += P * (np.dot(espqval, espqval) / self.nesp - (np.sum(espqval) / self.nesp)**2)
            if AGrad:
                dVdqM   = (dVdqP * dqPdqM).T
                for p, vsd in ddVdqPdVS.items():
                    dVdqM[p,:] += flat(vsd[i] * col(getqatoms(mvals)))
                G      += flat(P * 2 * dVdqM * col(desp)) / self.nesp
                if AHess:
                    d2VdqM2 = np.zeros(dVdqM.shape)
                    for p, vsd in dddVdqPdVS2.items():
                        d2VdqM2[p,:] += flat(vsd[i] * col(getqatoms(mvals)))
                    H      += np.array(P * 2 * (dVdqM * dVdqM.T + d2VdqM2 * col(desp))) / self.nesp
        # Redundant but we keep it anyway
        D /= Z
        X /= Z
        X /= D
        G /= Z
        G /= D
        H /= Z
        H /= D
        if not in_fd():
            self.esp_err = np.sqrt(X)
        # Following is the restraint part
        # RESP hyperbola "strength" parameter; 0.0005 is weak, 0.001 is strong
        # RESP hyperbola "tightness" parameter; don't need to change this
        a = self.resp_a
        b = self.resp_b
        q = getqatoms(mvals)
        R   = a*np.sum((q**2 + b**2)**0.5 - b)
        dR  = a*q*(q**2 + b**2)**-0.5
        ddR = a*b**2*(q**2 + b**2)**-1.5
        self.respterm = R
        X += R
        if AGrad:
            G += flat(dqPdqM.T * col(dR))
            if AHess:
                H += np.diag(flat(dqPdqM.T * col(ddR)))
            
        Answer = {'X':X,'G':G,'H':H}
        return Answer

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        tw = self.w_energy + self.w_force + self.w_netforce + self.w_torque + self.w_resp
        if tw > 0.0:
            w_ef = (self.w_energy + self.w_force + self.w_netforce + self.w_torque) / tw
            w_resp = self.w_resp / tw
        else:
            w_ef = 0.0
            w_resp = 0.0
        if self.energy or self.force:
            Answer_EF = self.get_energy_force(mvals, AGrad, AHess)
            for i in Answer_EF:
                Answer[i] += w_ef * Answer_EF[i]
        if self.resp:
            Answer_ESP = self.get_resp(mvals, AGrad, AHess)
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
    X2      = np.dot(XiZ.flatten(),WCiW.flatten())
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
    X2      = np.dot(XiZ.flatten(),WCiW.flatten())
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
    X2      = np.dot(XiZ.flatten(),WCiW.flatten())
    return X2

