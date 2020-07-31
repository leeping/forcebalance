""" @package forcebalance.torsion_profile Torsion profile fitting module.

@author Lee-Ping Wang
@date 08/2019
"""
from __future__ import division
import os
import numpy as np
import itertools
import json
from copy import deepcopy
from collections import OrderedDict
from forcebalance.nifty import eqcgmx, printcool, printcool_dictionary, warn_press_key
from forcebalance.target import Target
from forcebalance.molecule import Molecule
from forcebalance.finite_difference import fdwrap, f12d3p, in_fd
from forcebalance.output import getLogger
from forcebalance.optimizer import Counter
logger = getLogger(__name__)

class TorsionProfileTarget(Target):
    """ Subclass of Target for fitting MM optimized geometries to QM optimized geometries. """
    def __init__(self, options, tgt_opts, forcefield):
        super(TorsionProfileTarget, self).__init__(options, tgt_opts, forcefield)

        if not os.path.exists(os.path.join(self.tgtdir, 'metadata.json')):
            raise RuntimeError('TorsionProfileTarget needs torsion drive metadata.json file')
        with open(os.path.join(self.tgtdir, 'metadata.json')) as f:
            self.metadata = json.load(f)
            self.ndim = len(self.metadata['dihedrals'])
            self.freeze_atoms = sorted(set(itertools.chain(*self.metadata['dihedrals'])))

        ## Read in the coordinate files and get topology information from PDB
        if hasattr(self, 'pdb') and self.pdb is not None:
            self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords),
                                top=(os.path.join(self.root,self.tgtdir,self.pdb)))
        else:
            self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords))
        ## Number of snapshots.
        self.ns = len(self.mol)
        ## Option for how much data to write to disk.
        self.set_option(tgt_opts,'writelevel','writelevel')
        ## Harmonic restraint for non-torsion atoms in kcal/mol.
        self.set_option(tgt_opts,'restrain_k','restrain_k')
        ## Attenuate the weights as a function of energy
        self.set_option(tgt_opts,'attenuate','attenuate')
        ## Energy denominator for objective function
        self.set_option(tgt_opts,'energy_denom','energy_denom')
        ## Set upper cutoff energy
        self.set_option(tgt_opts,'energy_upper','energy_upper')
        ## Read in the reference data.
        self.read_reference_data()
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(list(self.OptionDict.items()) + list(options.items()))
        engine_args.pop('name', None)
        ## Create engine object.
        engine_args['freeze_atoms'] = self.freeze_atoms
        self.engine = self.engine_(target=self, mol=self.mol, **engine_args)

    def read_reference_data(self):

        """ Read the reference ab initio data from a file such as qdata.txt.

        After reading in the information from qdata.txt, it is converted
        into the GROMACS energy units (kind of an arbitrary choice).
        """
        ## Reference (QM) energies
        self.eqm           = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.tgtdir,"qdata.txt")
        # Parse the qdata.txt file
        for line in open(os.path.join(self.root,self.qfnm)):
            sline = line.split()
            if len(sline) == 0: continue
            elif sline[0] == 'ENERGY':
                self.eqm.append(float(sline[1]))

        if len(self.eqm) != self.ns:
            raise RuntimeError("Length of qdata.txt should match number of structures")

        # Turn everything into arrays, convert to kcal/mol
        self.eqm = np.array(self.eqm)
        self.eqm *= eqcgmx / 4.184
        # Use the minimum energy structure of the QM as reference
        self.eqm  -= np.min(self.eqm)
        self.smin  = np.argmin(self.eqm)
        logger.info("Referencing all energies to the snapshot %i (minimum energy structure in QM)\n" % self.smin)

        if self.attenuate:
            # Attenuate energies by an amount proportional to their
            # value above the minimum.
            eqm1 = self.eqm - np.min(self.eqm)
            denom = self.energy_denom
            upper = self.energy_upper
            self.wts = np.ones(self.ns)
            for i in range(self.ns):
                if eqm1[i] > upper:
                    self.wts[i] = 0.0
                elif eqm1[i] < denom:
                    self.wts[i] = 1.0 / denom
                else:
                    self.wts[i] = 1.0 / np.sqrt(denom**2 + (eqm1[i]-denom)**2)
        else:
            self.wts = np.ones(self.ns)

        # Normalize weights.
        self.wts /= sum(self.wts)

    def indicate(self):
        title_str = "Torsion Profile: %s, Objective = % .5e, Units = kcal/mol, Angstrom" % (self.name, self.objective)
        #LPW: This title is carefully placed to align correctly
        column_head_str1 =  "%-50s %-10s %-12s %-18s %-12s %-10s %-11s %-10s" % ("System", "Min(QM)", "Min(MM)", "Range(QM)", "Range(MM)", "Max-RMSD", "Ene-RMSE", "Obj-Fn")
        printcool_dictionary(self.PrintDict,title=title_str + '\n' + column_head_str1, keywidth=50, center=[True,False])

    def get(self, mvals, AGrad=False, AHess=False):
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        self.PrintDict = OrderedDict()

        def compute(mvals_, indicate=False):
            self.FF.make(mvals_)
            M_opts = None
            compute.emm = []
            compute.rmsd = []
            for i in range(self.ns):
                energy, rmsd, M_opt = self.engine.optimize(shot=i, align=False)
                # Create a molecule object to hold the MM-optimized structures
                compute.emm.append(energy)
                compute.rmsd.append(rmsd)
                if M_opts is None:
                    M_opts = deepcopy(M_opt)
                else:
                    M_opts += M_opt
            compute.emm = np.array(compute.emm)
            compute.emm -= compute.emm[self.smin]
            compute.rmsd = np.array(compute.rmsd)
            if indicate:
                if self.writelevel > 0:
                    energy_comparison = np.array([
                        self.eqm,
                        compute.emm,
                        compute.emm - self.eqm,
                        np.sqrt(self.wts)/self.energy_denom
                    ]).T
                    np.savetxt("EnergyCompare.txt", energy_comparison, header="%11s  %12s  %12s  %12s" % ("QMEnergy", "MMEnergy", "Delta(MM-QM)", "Weight"), fmt="% 12.6e")
                    M_opts.write('mm_minimized.xyz')
                    if self.ndim == 1:
                        try:
                            import matplotlib.pyplot as plt
                            plt.switch_backend('agg')
                            fig, ax = plt.subplots()
                            dihedrals = np.array([i[0] for i in self.metadata['torsion_grid_ids']])
                            dsort = np.argsort(dihedrals)
                            ax.plot(dihedrals[dsort], self.eqm[dsort], label='QM')
                            if hasattr(self, 'emm_orig'):
                                ax.plot(dihedrals[dsort], compute.emm[dsort], label='MM Current')
                                ax.plot(dihedrals[dsort], self.emm_orig[dsort], label='MM Initial')
                            else:
                                ax.plot(dihedrals[dsort], compute.emm[dsort], label='MM Initial')
                                self.emm_orig = compute.emm.copy()
                            ax.legend()
                            ax.set_xlabel('Dihedral (degree)')
                            ax.set_ylabel('Energy (kcal/mol)')
                            fig.suptitle('Torsion profile: iteration %i\nSystem: %s' % (Counter(), self.name))
                            fig.savefig('plot_torsion.pdf')
                        except ImportError:
                            logger.warning("matplotlib package is needed to make torsion profile plots\n")
            return (np.sqrt(self.wts)/self.energy_denom) * (compute.emm - self.eqm)
        compute.emm = None
        compute.rmsd = None

        V = compute(mvals, indicate=True)

        Answer['X'] = np.dot(V,V)

        # Energy RMSE
        e_rmse = np.sqrt(np.dot(self.wts, (compute.emm - self.eqm)**2))

        self.PrintDict[self.name] = '%10s %10s    %6.3f - %-6.3f   % 6.3f - %-6.3f    %6.3f    %7.4f   % 7.4f' % (','.join(['%i' % i for i in self.metadata['torsion_grid_ids'][self.smin]]),
                                                                                                                  ','.join(['%i' % i for i in self.metadata['torsion_grid_ids'][np.argmin(compute.emm)]]),
                                                                                                                  min(self.eqm), max(self.eqm), min(compute.emm), max(compute.emm), max(compute.rmsd), e_rmse, Answer['X'])

        # compute gradients and hessian
        dV = np.zeros((self.FF.np,len(V)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(compute, mvals, p), h = self.h, f0 = V)

        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(V, dV[p,:])
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:])
        if not in_fd():
            self.objective = Answer['X']
            self.FF.make(mvals)
        return Answer
