""" @package forcebalance.abinitio Ab-initio fitting module (energies, forces, resp).

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, warn_press_key, warn_once, pvec1d, commadash, uncommadash, isint
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

def norm2(arr, a=0, n=None, step=3):
    """
    Given a one-dimensional array, return the norm-squared of
    every "step" elements, starting at 'a' and computing 'n' total
    elements (so arr[a:a+step*n] must be valid).

    Parameters
    ----------
    arr : np.ndarray
        One-dimensional array to be normed
    a : int, default=0
        The starting index
    n : int, or None
        The number of norms to calculate (in intervals of step)
    step : int, default=3
        The number of elements in each norm calculation (this is usually 3)
    """
    if len(arr.shape) != 1:
        raise RuntimeError("Please only pass a one-dimensional array")
    if n is not None:
        if arr.shape[0] < a+step*n:
            raise RuntimeError("Please provide an array of length >= %i" % (a+step*n))
    else:
        if ((arr.shape[0]-a)%step != 0):
            raise RuntimeError("Please provide an array with (length-%i) divisible by %i" % (a, step))
        n = (arr.shape[0]-a)/step
    answer = []
    for j in range(n):
        d = arr[a+step*j:a+step*j+step]
        answer.append(np.dot(d,d))
    return np.array(answer)

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
        ## Whether to match Absolute Energies (make sure you know what you're doing)
        self.set_option(tgt_opts,'absolute','absolute')
        ## Number of atoms that we are fitting
        self.set_option(tgt_opts,'fitatoms','fitatoms_in')
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
        if not self.force:
            self.w_force = 0.0
        self.set_option(tgt_opts,'force_map','force_map')
        self.set_option(tgt_opts,'w_netforce','w_netforce')
        self.set_option(tgt_opts,'w_torque','w_torque')
        self.set_option(tgt_opts,'w_resp','w_resp')
        # Normalize the contributions to the objective function
        self.set_option(tgt_opts,'w_normalize')
        ## Option for how much data to write to disk.
        self.set_option(tgt_opts,'writelevel','writelevel')
        ## Whether to do energy and force calculations for the whole trajectory, or to do
        ## one calculation per snapshot.
        self.set_option(tgt_opts,'all_at_once','all_at_once')
        ## OpenMM-only option - whether to run the energies and forces internally.
        self.set_option(tgt_opts,'run_internal','run_internal')
        ## Whether we have virtual sites (set at the global option level)
        self.set_option(options,'have_vsite','have_vsite')
        ## Attenuate the weights as a function of energy
        self.set_option(tgt_opts,'attenuate','attenuate')
        ## What is the energy denominator? (Valid for 'attenuate')
        self.set_option(tgt_opts,'energy_denom','energy_denom')
        ## Set upper cutoff energy
        self.set_option(tgt_opts,'energy_upper','energy_upper')
        ## Assign a greater weight to MM snapshots that underestimate the QM energy (surfaces referenced to QM absolute minimum)
        self.set_option(tgt_opts,'energy_asymmetry')
        self.savg = (self.energy_asymmetry == 1.0 and not self.absolute)
        self.asym = (self.energy_asymmetry != 1.0)
        if self.asym:
            if not self.all_at_once:
                logger.error("Asymmetric weights only work when all_at_once is enabled")
                raise RuntimeError
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Boltzmann weights
        self.boltz_wts = []
        ## Reference (QM) energies
        self.eqm           = []
        ## Reference (QM) forces
        self.fqm           = []
        ## ESP grid points
        self.espxyz        = []
        ## ESP values
        self.espval        = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.tgtdir,"qdata.txt")
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
        self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords),
                            top=(os.path.join(self.root,self.tgtdir,self.pdb) if hasattr(self, 'pdb') else None))
        ## Set the number of snapshots
        if self.ns != -1:
            self.mol = self.mol[:self.ns]
        self.ns = len(self.mol)
        ## The number of (atoms + drude particles + virtual sites)
        self.nparticles  = len(self.mol.elem)
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(self.OptionDict.items() + options.items())
        del engine_args['name']
        ## Create engine object.
        self.engine = self.engine_(target=self, mol=self.mol, **engine_args)
        ## Lists of atoms to do net force/torque fitting and excluding virtual sites.
        self.AtomLists = self.engine.AtomLists
        self.AtomMask = self.engine.AtomMask
        ## Read in the reference data
        self.read_reference_data()
        ## The below two options are related to whether we want to rebuild virtual site positions.
        ## Rebuild the distance matrix if virtual site positions have changed
        self.buildx = True
        ## Save the mvals from the last time we updated the vsites.
        self.save_vmvals = {}
        self.set_option(None, 'shots', val=self.ns)

    def build_invdist(self, mvals):
        for i in self.pgrad:
            if 'VSITE' in self.FF.plist[i]:
                if i in self.save_vmvals and mvals[i] != self.save_vmvals[i]:
                    self.buildx = True
                    break
        if not self.buildx: return self.invdists
        if any(['VSITE' in i for i in self.FF.map.keys()]) or self.have_vsite:
            logger.info("\rGenerating virtual site positions.%s" % (" "*30))
            pvals = self.FF.make(mvals)
            self.mol.xyzs = self.engine.generate_positions()
        # prepare the distance matrix for esp computations
        if len(self.espxyz) > 0:
            invdists = []
            logger.info("\rPreparing the distance matrix... it will have %i * %i * %i = %i elements" % (self.ns, self.nesp, self.nparticles, self.ns * self.nesp * self.nparticles))
            sn = 0
            for espset, xyz in zip(self.espxyz, self.mol.xyzs):
                logger.info("\rGenerating ESP distances for snapshot %i%s\r" % (sn, " "*50))
                esparr = np.array(espset).reshape(-1,3)
                # Create a matrix with Nesp rows and Natoms columns.
                DistMat = np.array([[np.linalg.norm(i - j) for j in xyz] for i in esparr])
                invdists.append(1. / (DistMat / bohrang))
                sn += 1
        for i in self.pgrad:
            if 'VSITE' in self.FF.plist[i]:
                self.save_vmvals[i] = mvals[i]
        self.buildx = False
        return np.array(invdists)

    def compute_netforce_torque(self, xyz, force, QM=False):
        # Convert an array of (3 * n_atoms) atomistic forces
        # to an array of (3 * (n_forces + n_torques)) net forces and torques.
        # This code is rather slow.  It requires the system to have a list
        # of masses and blocking numbers.

        kwds = {"MoleculeNumber" : "molecule",
                "ResidueNumber" : "residue",
                "ChargeGroupNumber" : "chargegroup"}
        if self.force_map == 'molecule' and 'MoleculeNumber' in self.AtomLists:
            Block = self.AtomLists['MoleculeNumber']
        elif self.force_map == 'residue' and 'ResidueNumber' in self.AtomLists:
            Block = self.AtomLists['ResidueNumber']
        elif self.force_map == 'chargegroup' and 'ChargeGroupNumber' in self.AtomLists:
            Block = self.AtomLists['ChargeGroupNumber']
        else:
            logger.error('force_map keyword "%s" is invalid. Please choose from: %s\n' % (self.force_map, ', '.join(['"%s"' % kwds[k] for k in self.AtomLists.keys() if k in kwds])))
            raise RuntimeError

        nft = len(self.fitatoms)
        # Number of particles that the force is acting on
        nfp = force.reshape(-1,3).shape[0]
        # Number of particles in the XYZ coordinates
        nxp = xyz.shape[0]
        # Number of particles in self.AtomLists
        npr = len(self.AtomMask)
        # Number of true atoms
        nat = sum(self.AtomMask)

        mask = np.array([i for i in range(npr) if self.AtomMask[i]])

        if nfp not in [npr, nat]:
            logger.error('Force contains %i particles but expected %i or %i\n' % (nfp, npr, nat))
            raise RuntimeError
        elif nfp == nat:
            frc1 = force.reshape(-1,3)[:nft].flatten()
        elif nfp == npr:
            frc1 = force.reshape(-1,3)[mask][:nft].flatten()

        if nxp not in [npr, nat]:
            logger.error('Coordinates contains %i particles but expected %i or %i\n' % (nfp, npr, nat))
            raise RuntimeError
        elif nxp == nat:
            xyz1 = xyz[:nft]
        elif nxp == npr:
            xyz1 = xyz[mask][:nft]

        Block = list(np.array(Block)[mask])[:nft]
        Mass = np.array(self.AtomLists['Mass'])[mask][:nft]

        NetForces = []
        Torques = []
        for b in sorted(set(Block)):
            AtomBlock = np.array([i for i in range(len(Block)) if Block[i] == b])
            CrdBlock = np.array(list(itertools.chain(*[range(3*i, 3*i+3) for i in AtomBlock])))
            com = np.sum(xyz1[AtomBlock]*np.outer(Mass[AtomBlock],np.ones(3)), axis=0) / np.sum(Mass[AtomBlock])
            frc = frc1[CrdBlock].reshape(-1,3)
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

        After reading in the information from qdata.txt, it is converted
        into the GROMACS energy units (kind of an arbitrary choice);
        forces (kind of a misnomer in qdata.txt) are multipled by -1
        to convert gradients to forces.

        We typically subtract out the mean energies of all energy arrays
        because energy/force matching does not account for zero-point
        energy differences between MM and QM (i.e. energy of electrons
        in core orbitals).

        A 'hybrid' ensemble is possible where we use 50% MM and 50% QM
        weights.  Please read more in LPW and Troy Van Voorhis, JCP
        Vol. 133, Pg. 231101 (2010), doi:10.1063/1.3519043.  In the
        updated version of the code (July 13 2016), this feature is
        currently not implemented due to disuse, but it is easy to
        re-implement if desired.

        Finally, note that using non-Boltzmann weights degrades the
        statistical information content of the snapshots.  This
        problem will generally become worse if the ensemble to which
        we're reweighting is dramatically different from the one we're
        sampling from.  We end up with a set of Boltzmann weights like
        [1e-9, 1e-9, 1.0, 1e-9, 1e-9 ... ] and this is essentially just
        one snapshot.

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
            elif sline[0] == 'FORCES':
                self.fqm.append([float(i) for i in sline[1:]])
            elif sline[0] == 'ESPXYZ':
                self.espxyz.append([float(i) for i in sline[1:]])
            elif sline[0] == 'ESPVAL':
                self.espval.append([float(i) for i in sline[1:]])

        # Ensure that all lists are of length self.ns
        self.eqm = self.eqm[:self.ns]
        self.fqm = self.fqm[:self.ns]
        self.espxyz = self.espxyz[:self.ns]
        self.espval = self.espval[:self.ns]

        # Turn everything into arrays, convert to kJ/mol, and subtract the mean energy from the energy arrays
        self.eqm = np.array(self.eqm)
        self.eqm *= eqcgmx
        if self.asym:
            self.eqm  -= np.min(self.eqm)
            self.smin  = np.argmin(self.eqm)
            logger.info("Referencing all energies to the snapshot %i (minimum energy structure in QM)\n" % self.smin)
        elif self.absolute:
            logger.info("Fitting absolute energies.  Make sure you know what you are doing!\n")
        else:
            self.eqm -= np.mean(self.eqm)

        if len(self.fqm) > 0:
            self.fqm = np.array(self.fqm)
            self.fqm *= fqcgmx
            self.qmatoms = range(self.fqm.shape[1]/3)
        else:
            logger.info("QM forces are not present, only fitting energies.\n")
            self.force = 0
            self.w_force = 0

        # Here we may choose a subset of atoms to do the force matching.
        if self.force:
            # Build a list corresponding to the atom indices where we are fitting the forces.
            if isint(self.fitatoms_in):
                if int(self.fitatoms_in) == 0:
                    self.fitatoms = self.qmatoms
                else:
                    warn_press_key("Provided an integer for fitatoms; will assume this means the first %i atoms" % int(self.fitatoms_in))
                    self.fitatoms = range(int(self.fitatoms_in))
            else:
                # If provided a "comma and dash" list, then expand the list.
                self.fitatoms = uncommadash(self.fitatoms_in)

            if len(self.fitatoms) > len(self.qmatoms):
                warn_press_key("There are more fitting atoms than the total number of atoms in the QM calculation (something is probably wrong)")
            else:
                if self.w_force > 0:
                    if len(self.fitatoms) == len(self.qmatoms):
                        logger.info("Fitting the forces on all atoms\n")
                    else:
                        logger.info("Fitting the forces on atoms %s\n" % commadash(self.fitatoms))
                        logger.info("Pruning the quantum force matrix...\n")
                selct = list(itertools.chain(*[[3*i+j for j in range(3)] for i in self.fitatoms]))
                self.fqm  = self.fqm[:, selct]
        else:
            self.fitatoms = []

        self.nesp = len(self.espval[0]) if len(self.espval) > 0 else 0

        if self.attenuate:
            # Attenuate energies by an amount proportional to their
            # value above the minimum.
            eqm1 = self.eqm - np.min(self.eqm)
            denom = self.energy_denom * 4.184 # kcal/mol to kJ/mol
            upper = self.energy_upper * 4.184 # kcal/mol to kJ/mol
            self.boltz_wts = np.ones(self.ns)
            for i in range(self.ns):
                if eqm1[i] > upper:
                    self.boltz_wts[i] = 0.0
                elif eqm1[i] < denom:
                    self.boltz_wts[i] = 1.0 / denom
                else:
                    self.boltz_wts[i] = 1.0 / np.sqrt(denom**2 + (eqm1[i]-denom)**2)
        else:
            self.boltz_wts = np.ones(self.ns)

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
            self.nnf = 0
            self.ntq = 0

        # Normalize Boltzmann weights.
        self.boltz_wts /= sum(self.boltz_wts)

    def indicate(self):
        Headings = ["Observable", "Difference\n(Calc-Ref)", "Denominator\n RMS (Ref)", " Percent \nDifference", "Term", "x Wt =", "Contrib."]
        Data = OrderedDict([])
        if self.energy:
            Data['Energy (kJ/mol)'] = ["%8.4f" % self.e_err,
                                       "%8.4f" % self.e_ref,
                                       "%.4f%%" % (self.e_err_pct*100),
                                       "%8.4f" % self.e_trm,
                                       "%.3f" % self.w_energy,
                                       "%8.4f" % self.e_ctr]
        if self.force:
            Data['Gradient (kJ/mol/A)'] = ["%8.4f" % (self.f_err/10),
                                           "%8.4f" % (self.f_ref/10),
                                           "%.4f%%" % (self.f_err_pct*100),
                                           "%8.4f" % self.f_trm,
                                           "%.3f" % self.w_force,
                                           "%8.4f" % self.f_ctr]
            if self.use_nft:
                Data['Net Force (kJ/mol/A)'] = ["%8.4f" % (self.nf_err/10),
                                                "%8.4f" % (self.nf_ref/10),
                                                "%.4f%%" % (self.nf_err_pct*100),
                                                "%8.4f" % self.nf_trm,
                                                "%.3f" % self.w_netforce,
                                                "%8.4f" % self.nf_ctr]
                Data['Torque (kJ/mol/rad)'] = ["%8.4f" % self.tq_err,
                                               "%8.4f" % self.tq_ref,
                                               "%.4f%%" % (self.tq_err_pct*100),
                                               "%8.4f" % self.tq_trm,
                                               "%.3f" % self.w_torque,
                                               "%8.4f" % self.tq_ctr]
        if self.resp:
            Data['Potential (a.u.'] = ["%8.4f" % (self.esp_err/10),
                                       "%8.4f" % (self.esp_ref/10),
                                       "%.4f%%" % (self.esp_err_pct*100),
                                       "%8.4f" % self.esp_trm,
                                       "%.3f" % self.w_resp,
                                       "%8.4f" % self.esp_ctr]
        self.printcool_table(data=Data, headings=Headings, color=0)
        if self.force:
            logger.info("Maximum force difference on atom %i (%s), frame %i, %8.4f kJ/mol/A\n" % (self.maxfatom, self.mol.elem[self.fitatoms[self.maxfatom]], self.maxfshot, self.maxdf/10))

    def energy_all(self):
        if hasattr(self, 'engine'):
            return self.engine.energy().reshape(-1,1)
        else:
            logger.error("Target must contain an engine object\n")
            raise NotImplementedError

    def energy_force_all(self):
        if hasattr(self, 'engine'):
            return self.engine.energy_force()
        else:
            logger.error("Target must contain an engine object\n")
            raise NotImplementedError

    def energy_force_transform(self):
        if self.force:
            M = self.energy_force_all()
            selct = [0] + list(itertools.chain(*[[1+3*i+j for j in range(3)] for i in self.fitatoms]))
            M = M[:, selct]
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
            logger.error("Target must contain an engine object\n")
            raise NotImplementedError

    def energy_force_one(self, i):
        if hasattr(self, 'engine'):
            return self.engine.energy_force_one(i)
        else:
            logger.error("Target must contain an engine object\n")
            raise NotImplementedError

    def energy_force_transform_one(self,i):
        if self.force:
            M = self.energy_force_one(i)
            selct = [0] + list(itertools.chain(*[[1+3*i+j for j in range(3)] for i in self.fitatoms]))
            M = M[:, selct]
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
        LPW 7-13-2016

        This code computes the least squares objective function for the energy and force.
        The most recent revision simplified the code to make it easier to maintain,
        and to remove the covariance matrix and dual-weight features.

        The equations have also been simplified.  Previously I normalizing each force
        component (or triples of components belonging to an atom) separately.
        I was also computing the objective function separately from the "indicators",
        which led to confusion regarding why they resulted in different values.
        The code was structured around computing the objective function by multiplying
        an Applequist polytensor containing both energy and force with an inverse
        covariance matrix, but it became apparent over time and experience that the
        more complicated approach was not worth it, given the opacity that it introduced
        into how things were computed.

        In the new code, the objective function is computed in a simple way.
        For the energies we compute a weighted sum of squared differences
        between E_MM and E_QM, minus (optionally) the mean energy gap, for the numerator.
        The weighted variance of the QM energies <E_QM^2>-<E_QM>^2 is the denominator.
        The user-supplied w_energy option is a prefactor that multiplies this term.
        If w_normalize is set to True (no longer the default), the prefactor is
        further divided by the sum of all of the weights.
        The indicators are set to the square roots of the numerator and denominator above.

        For the forces we compute the same weighted sum, where each term in the sum
        is the norm-squared of F_MM - F_QM, with no option to subtract the mean.
        The denominator is computed in an analogous way using the norm-squared F_QM,
        and the prefactor is w_force. The same approach is applied if the user asks
        for net forces and/or torques. The indicators are computed from the square
        roots of the numerator and denominator, divided by the number of
        atoms (molecules) for forces (net forces / torques).

        In equation form, the objective function is given by:

        \[ = {W_E}\left[ {\frac{{\left( {\sum\limits_{i \in {N_s}}
        {{w_i}{{\left( {E_i^{MM} - E_i^{QM}} \right)}^2}} } \right) -
        {{\left( {{{\bar E}^{MM}} - {{\bar E}^{QM}}} \right)}^2}}}
        {{\sum\limits_{i \in {N_s}} {{w_i}{{\left(
        {E_i^{QM} - {{\bar E}^{QM}}} \right)}^2}} }}} \right] +
        {W_F}\left[ {\frac{{\sum\limits_{i \in {N_s}} {{w_i}\sum\limits_{a \in {N_a}}
        {{{\left| {{\bf{F}}_{i,a}^{MM} - {\bf{F}}_{i,a}^{QM}} \right|}^2}} } }}
        {{\sum\limits_{i \in {N_s}} {{w_i}\sum\limits_{a \in {N_a}}
        {{{\left| {{\bf{F}}_{i,a}^{QM}} \right|}^2}} } }}} \right]\]

        In the previous code (ForTune, 2011 and previous)
        this subroutine used analytic first derivatives of the
        energy and force to build the derivatives of the objective function.
        Here I will take a simplified approach, because building the analytic
        derivatives require substantial modifications of the engine code,
        which is unsustainable. We use a finite different calculation
        of the first derivatives of the energies and forces to get the exact
        first derivative and approximate second derivative of the objective function..

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
        NC   = 3*nat
        NCP1 = 3*nat+1
        NParts = 1
        if self.force:
            NParts += 1
        if self.use_nft:
            NParts += 2
            NCP1 += 3*(nnf + ntq)
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
            if self.asym:
                M_all[:, 0] -= M_all[self.smin, 0]
            if AGrad or AHess:
                def callM(mvals_):
                    logger.debug("\r")
                    pvals = self.FF.make(mvals_)
                    return self.energy_force_transform()
                for p in self.pgrad:
                    dM_all[:,p,:], ddM_all[:,p,:] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M_all)
                    if self.asym:
                        dM_all[:, p, 0] -= dM_all[self.smin, p, 0]
                        ddM_all[:, p, 0] -= ddM_all[self.smin, p, 0]
        if self.force and not in_fd():
            self.maxfatom = -1
            self.maxfshot = -1
            self.maxdf = 0.0
        for i in range(NS):
            if i % 100 == 0:
                logger.debug("\rIncrementing quantities for snapshot %i\r" % i)
            # Build Boltzmann weights and increment partition function.
            P   = self.boltz_wts[i]
            Z  += P
            # Load reference (QM) data
            Q[0] = self.eqm[i]
            if self.force:
                Q[1:] = self.fref[i,:].copy()
            QQ     = Q*Q
            # Call the simulation software to get the MM quantities
            # (or load from M_all array)
            if self.all_at_once:
                M = M_all[i]
            else:
                if i % 100 == 0:
                    logger.debug("Shot %i\r" % i)
                M = self.energy_force_transform_one(i)
                M_all[i,:] = M.copy()
            # MM - QM difference
            X     = M-Q
            boost   = 1.0
            # For asymmetric fit, MM energies lower than QM are given a boost factor
            if self.asym and X[0] < 0.0:
                boost = self.energy_asymmetry
            # Save information about forces
            if self.force:
                # Norm-squared of force differences for each atom
                dfrc2 = norm2(M-Q, 1, nat)
                if not in_fd() and np.max(dfrc2) > self.maxdf:
                    self.maxdf = np.sqrt(np.max(dfrc2))
                    self.maxfatom = np.argmax(dfrc2)
                    self.maxfshot = i
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
                if self.all_at_once:
                    M_p[p] = dM_all[i, p]
                    M_pp[p] = ddM_all[i, p]
                else:
                    def callM(mvals_):
                        if i % 100 == 0:
                            logger.debug("\r")
                        pvals = self.FF.make(mvals_)
                        return self.energy_force_transform_one(i)
                    M_p[p],M_pp[p] = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = M)
                if all(M_p[p] == 0): continue
                M0_p[p][0]  += P * M_p[p][0]
                Xi_p        = 2 * X * M_p[p]
                Xi_p[0]    *= boost
                SPX_p[p] += P * Xi_p
                if not AHess: continue
                if self.all_at_once:
                    M_pp[p] = ddM_all[i, p]
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
        if self.savg:
            M_all_print[:,0] -= np.average(M_all_print[:,0], weights=self.boltz_wts)
        if self.force:
            Q_all_print = np.hstack((col(self.eqm),self.fref))
        else:
            Q_all_print = col(self.eqm)
        if self.savg:
            QEtmp = np.array(Q_all_print[:,0]).flatten()
            Q_all_print[:,0] -= np.average(QEtmp, weights=self.boltz_wts)
        if self.attenuate:
            QEtmp = np.array(Q_all_print[:,0]).flatten()
            Q_all_print[:,0] -= np.min(QEtmp)
            MEtmp = np.array(M_all_print[:,0]).flatten()
            M_all_print[:,0] -= np.min(MEtmp)
        if self.writelevel > 1:
            np.savetxt('M.txt',M_all_print)
            np.savetxt('Q.txt',Q_all_print)
        if self.writelevel > 0:
            EnergyComparison = np.hstack((col(Q_all_print[:,0]),
                                          col(M_all_print[:,0]),
                                          col(M_all_print[:,0])-col(Q_all_print[:,0]),
                                          col(self.boltz_wts)))
            np.savetxt("EnergyCompare.txt", EnergyComparison, header="%11s  %12s  %12s  %12s" % ("QMEnergy", "MMEnergy", "Delta(MM-QM)", "Weight"), fmt="% 12.6e")
            plot_mm_vs_qm(M_all_print[:,0], Q_all_print[:,0], title='Abinitio '+self.name)
        if self.force and self.writelevel > 1:
            # Write .xyz files which can be viewed in vmd.
            QMTraj = self.mol[:].atom_select(self.fitatoms)
            Mforce_obj = QMTraj[:]
            Qforce_obj = QMTraj[:]
            Mforce_print = np.array(M_all_print[:,1:3*nat+1])
            Qforce_print = np.array(Q_all_print[:,1:3*nat+1])
            MaxComp = np.max(np.abs(np.vstack((Mforce_print,Qforce_print)).flatten()))
            Mforce_print /= MaxComp
            Qforce_print /= MaxComp
            for i in range(NS):
                Mforce_obj.xyzs[i] = Mforce_print[i, :].reshape(-1,3)
                Qforce_obj.xyzs[i] = Qforce_print[i, :].reshape(-1,3)
            # if nat < len(self.qmatoms):
            #     Fpad = np.zeros((len(self.qmatoms) - nat, 3))
            #     Mforce_obj.xyzs[i] = np.vstack((Mforce_obj.xyzs[i], Fpad))
            #     Qforce_obj.xyzs[i] = np.vstack((Qforce_obj.xyzs[i], Fpad))
            if Mforce_obj.na != Mforce_obj.xyzs[0].shape[0]:
                print Mforce_obj.na
                print Mforce_obj.xyzs[0].shape[0]
                warn_once('\x1b[91mThe printing of forces is not set up correctly.  Not printing forces.  Please report this issue.\x1b[0m')
            else:
                if self.writelevel > 1:
                    QMTraj.write('coords.xyz')
                    Mforce_obj.elem = ['H' for i in range(Mforce_obj.na)]
                    Mforce_obj.write('MMforce.xyz')
                    Qforce_obj.elem = ['H' for i in range(Qforce_obj.na)]
                    Qforce_obj.write('QMforce.xyz')

        #==============================================================#
        #            STEP 3: Build the objective function.             #
        #==============================================================#

        logger.debug("Done with snapshots, building objective function now\r")
        W_Components = np.zeros(NParts)
        W_Components[0] = self.w_energy
        if self.force:
            W_Components[1] = self.w_force
            if self.use_nft:
                W_Components[2] = self.w_netforce
                W_Components[3] = self.w_torque
        if np.sum(W_Components) > 0 and self.w_normalize:
            W_Components /= np.sum(W_Components)

        def compute_objective(SPX_like,divide=1,L=None,R=None,L2=None,R2=None):
            a = 0
            n = 1
            X2E = compute_objective_part(SPX_like,QQ0,Q0,Z,a,n,energy=True,subtract_mean=self.savg,
                                         divide=divide,L=L,R=R,L2=L2,R2=R2)
            objs = [X2E]
            if self.force:
                a = 1
                n = 3*nat
                X2F = compute_objective_part(SPX_like,QQ0,Q0,Z,a,n,divide=divide)
                objs.append(X2F)
                if self.use_nft:
                    a += n
                    n = 3*nnf
                    X2N = compute_objective_part(SPX_like,QQ0,Q0,Z,a,n,divide=divide)
                    objs.append(X2N)
                    a += n
                    n = 3*ntq
                    X2T = compute_objective_part(SPX_like,QQ0,Q0,Z,a,n,divide=divide)
                    objs.append(X2T)
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
            if self.force:
                self.f_trm = X2_Components[1]
                self.f_ctr = X2_Components[1]*W_Components[1]
                if self.w_normalize: self.f_ctr /= tw
                self.f_ref = np.sqrt(X2_Normalize[1]/nat)
                self.f_err = np.sqrt(X2_Physical[1]/nat)
                self.f_err_pct = self.f_err/self.f_ref
            if self.use_nft:
                self.nf_trm = X2_Components[2]
                self.nf_ctr = X2_Components[2]*W_Components[2]
                self.nf_ref = np.sqrt(X2_Normalize[2]/nnf)
                self.nf_err = np.sqrt(X2_Physical[2]/nnf)
                self.nf_err_pct = self.nf_err/self.nf_ref
                self.tq_trm = X2_Components[3]
                self.tq_ctr = X2_Components[3]*W_Components[3]
                self.tq_ref = np.sqrt(X2_Normalize[3]/ntq)
                self.tq_err = np.sqrt(X2_Physical[3]/ntq)
                self.tq_err_pct = self.tq_err/self.tq_ref
                if self.w_normalize:
                    self.nf_ctr /= tw
                    self.tq_ctr /= tw

            pvals = self.FF.make(mvals) # Write a force field that isn't perturbed by finite differences.
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
        def new_charges(mvals_):
            """ Return the charges acting on the system. """
            logger.debug("\r")
            pvals = self.FF.make(mvals_)
            return self.engine.get_charges()

            # Need to update the positions of atoms, if there are virtual sites.
            # qvals = [pvals[i] for i in self.FF.qmap]
            # # All of a sudden I need the number of virtual sites.
            # qatoms = np.zeros(self.nparticles)
            # for i, jj in enumerate(self.FF.qid):
            #     for j in jj:
            #         qatoms[j] = qvals[i]
            # return qatoms

        # Obtain a derivative matrix the stupid way
        charge0 = new_charges(mvals)
        if AGrad:
            # dqPdqM = []
            # for i in range(NP):
            #     print "Now working on parameter number", i
            #     dqPdqM.append(f12d3p(fdwrap(new_charges,mvals,i), h = self.h)[0])
            # dqPdqM = mat(dqPdqM).T
            dqPdqM = np.matrix([(f12d3p(fdwrap(new_charges,mvals,i), h = self.h, f0 = charge0)[0] if i in self.pgrad else np.zeros_like(charge0)) for i in range(NP)]).T
        xyzs = np.array(self.mol.xyzs)
        espqvals = np.array(self.espval)
        espxyz   = np.array(self.espxyz)

        ddVdqPdVS = {}
        # Second derivative of the inverse distance matrix with respect to the virtual site position
        dddVdqPdVS2 = {}
        if AGrad:
            for p in self.pgrad:
                if 'VSITE' in self.FF.plist[p]:
                    ddVdqPdVS[p], dddVdqPdVS2[p] = f12d3p(fdwrap(self.build_invdist,mvals,p), h = self.h, f0 = self.invdists)
        X = 0
        Q = 0
        D = 0
        G = np.zeros(NP)
        H = np.zeros((NP, NP))
        for i in range(self.ns):
            P   = self.boltz_wts[i]
            Z  += P
            dVdqP   = np.matrix(self.invdists[i])
            espqval = espqvals[i]
            espmval = dVdqP * col(new_charges(mvals))
            desp    = flat(espmval) - espqval
            X      += P * np.dot(desp, desp) / self.nesp
            Q      += P * np.dot(espqval, espqval) / self.nesp
            D      += P * (np.dot(espqval, espqval) / self.nesp - (np.sum(espqval) / self.nesp)**2)
            if AGrad:
                dVdqM   = (dVdqP * dqPdqM).T
                for p, vsd in ddVdqPdVS.items():
                    dVdqM[p,:] += flat(vsd[i] * col(new_charges(mvals)))
                G      += flat(P * 2 * dVdqM * col(desp)) / self.nesp
                if AHess:
                    d2VdqM2 = np.zeros(dVdqM.shape)
                    for p, vsd in dddVdqPdVS2.items():
                        d2VdqM2[p,:] += flat(vsd[i] * col(new_charges(mvals)))
                    H      += np.array(P * 2 * (dVdqM * dVdqM.T + d2VdqM2 * col(desp))) / self.nesp
        # Redundant but we keep it anyway
        D /= Z
        X /= Z
        X /= D
        G /= Z
        G /= D
        H /= Z
        H /= D
        Q /= Z
        Q /= D
        if not in_fd():
            self.esp_err = np.sqrt(X)
            self.esp_ref = np.sqrt(Q)
            self.esp_err_pct = self.esp_err / self.esp_ref

        # Following is the restraint part
        # RESP hyperbola "strength" parameter; 0.0005 is weak, 0.001 is strong
        # RESP hyperbola "tightness" parameter; don't need to change this
        a = self.resp_a
        b = self.resp_b
        q = new_charges(mvals)
        R   = a*np.sum((q**2 + b**2)**0.5 - b)
        dR  = a*q*(q**2 + b**2)**-0.5
        ddR = a*b**2*(q**2 + b**2)**-1.5
        self.respterm = R
        X += R
        if AGrad:
            G += flat(dqPdqM.T * col(dR))
            if AHess:
                H += np.diag(flat(dqPdqM.T * col(ddR)))

        if not in_fd():
            self.esp_trm = X
            self.esp_ctr = X*self.w_resp
            if self.w_normalize:
                tw = self.w_energy + self.w_force + self.w_netforce + self.w_torque + self.w_resp
                self.esp_ctr /= tw
        Answer = {'X':X,'G':G,'H':H}
        return Answer

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        tw = self.w_energy + self.w_force + self.w_netforce + self.w_torque + self.w_resp
        if self.w_normalize:
            w_ef /= tw
            w_resp = self.w_resp / tw
        else:
            w_ef = 1.0
            w_resp = self.w_resp
        if self.energy or self.force:
            Answer_EF = self.get_energy_force(mvals, AGrad, AHess)
            for i in Answer_EF:
                Answer[i] += w_ef * Answer_EF[i]
        if self.resp:
            Answer_ESP = self.get_resp(mvals, AGrad, AHess)
            for i in Answer_ESP:
                Answer[i] += w_resp * Answer_ESP[i]
        if not any([self.energy, self.force, self.resp]):
            logger.error("Ab initio fitting must have at least one of: Energy, Force, ESP\n")
            raise RuntimeError
        if not in_fd():
            self.objective = Answer['X']
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

def plot_mm_vs_qm(M, Q, title=''):
    import matplotlib.pyplot as plt
    qm_min_dx = np.argmin(Q)
    e_qm = Q - Q[qm_min_dx]
    e_mm = M - M[qm_min_dx]
    plt.plot(e_mm, e_qm, 'o')
    plt.xlabel('QM Energies (kJ/mol)')
    plt.ylabel('MM Energies (kJ/mol)')
    x1,x2,y1,y2 = plt.axis()
    if x2 < y2:
        x2 = y2
    else:
        y2 = x2
    plt.axis((0,x2,0,y2))
    plt.plot([0,x2],[0,y2], '--' )
    plt.title(title)
    plt.savefig('e_qm_vs_mm.pdf')
    plt.close()
