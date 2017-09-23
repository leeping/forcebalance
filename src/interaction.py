""" @package forcebalance.interaction Interaction energy fitting module.

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
import numpy as np
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, commadash, uncommadash, printcool_dictionary
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Interaction(Target):

    """ Subclass of Target for fitting force fields to interaction energies.

    Currently TINKER is supported.

    We introduce the following concepts:
    - The number of snapshots
    - The reference interaction energies and the file they belong in (qdata.txt)

    This subclass contains the 'get' method for building the objective
    function from any simulation software (a driver to run the program and
    read output is still required)."""

    def __init__(self,options,tgt_opts,forcefield):
        # Initialize the SuperClass!
        super(Interaction,self).__init__(options,tgt_opts,forcefield)

        #======================================#
        # Options that are given by the parser #
        #======================================#

        ## Number of snapshots
        self.set_option(tgt_opts,'shots','ns')
        ## Do we call Q-Chem for dielectric energies? (Currently needs to be fixed)
        self.set_option(tgt_opts,'do_cosmo','do_cosmo')
        ## Do we put the reference energy into the denominator?
        self.set_option(tgt_opts,'cauchy','cauchy')
        ## Do we put the reference energy into the denominator?
        self.set_option(tgt_opts,'attenuate','attenuate')
        ## Divide by the number of snapshots?
        self.set_option(tgt_opts, 'normalize')
        ## What is the energy denominator?
        self.set_option(tgt_opts,'energy_denom','energy_denom')
        ## Set fragment 1
        self.set_option(tgt_opts,'fragment1','fragment1')
        if len(self.fragment1) == 0:
            logger.error('You need to define the first fragment using the fragment1 keyword\n')
            raise RuntimeError
        self.select1 = np.array(uncommadash(self.fragment1))
        ## Set fragment 2
        self.set_option(tgt_opts,'fragment2','fragment2')
        if len(self.fragment2) != 0:
            self.select2 = np.array(uncommadash(self.fragment2))
        else:
            self.select2 = None
        ## Set upper cutoff energy
        self.set_option(tgt_opts,'energy_upper','energy_upper')
        ## Option for how much data to write to disk.
        self.set_option(tgt_opts,'writelevel','writelevel')
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Reference (QM) interaction energies
        self.eqm           = []
        ## Snapshot label, useful for graphing
        self.label         = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.tgtdir,"qdata.txt")
        self.e_err = 0.0
        self.e_err_pct = None
        ## Read in the trajectory file
        self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords),
                            top=(os.path.join(self.root,self.tgtdir,self.pdb) if hasattr(self, 'pdb') else None), build_topology=False if self.coords.endswith('.pdb') else True)
        if self.ns != -1:
            self.mol = self.mol[:self.ns]
        self.ns = len(self.mol)
        if self.select2 is None:
            self.select2 = [i for i in range(self.mol.na) if i not in self.select1]
            logger.info('Fragment 2 is the complement of fragment 1 : %s\n' % (commadash(self.select2)))
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(self.OptionDict.items() + options.items())
        del engine_args['name']
        self.engine = self.engine_(target=self, mol=self.mol, **engine_args)
        ## Read in the reference data
        self.read_reference_data()
        logger.info("The energy denominator is: %s kcal/mol\n"  % str(self.energy_denom))
        denom = self.energy_denom
        # Create the denominator.
        if self.cauchy:
            self.divisor = np.sqrt(self.eqm**2 + denom**2)
            if self.attenuate:
                logger.error('attenuate and cauchy are mutually exclusive\n')
                raise RuntimeError
        elif self.attenuate:
            # Attenuate only large repulsions.
            self.divisor = np.zeros(len(self.eqm))
            for i in range(len(self.eqm)):
                if self.eqm[i] < denom:
                    self.divisor[i] = denom
                else:
                    self.divisor[i] = np.sqrt(denom**2 + (self.eqm[i]-denom)**2)
        else:
            self.divisor = np.ones(len(self.eqm)) * denom

        if self.cauchy:
            logger.info("Each contribution to the interaction energy objective function will be scaled by 1.0 / ( energy_denom**2 + reference**2 )\n")
        if self.energy_upper > 0:
            ecut = self.energy_upper
            self.prefactor = 1.0 * (self.eqm < ecut)
            logger.info("Interactions more repulsive than %s will not be fitted (%i/%i excluded) \n" % (str(self.energy_upper), sum(self.eqm > ecut), len(self.eqm)))
        else:
            self.prefactor = np.ones(len(self.eqm))
        if self.normalize:
            self.prefactor /= len(self.prefactor)

    def read_reference_data(self):

        """ Read the reference ab initio data from a file such as qdata.txt.

        After reading in the information from qdata.txt, it is converted
        into kcal/mol.

        """
        # Parse the qdata.txt file
        for line in open(os.path.join(self.root,self.qfnm)):
            sline = line.split()
            if len(sline) == 0: continue
            elif sline[0] == 'INTERACTION':
                self.eqm.append(float(sline[1]))
            elif sline[0] == 'LABEL':
                self.label.append(sline[1])
            if all(len(i) in [self.ns, 0] for i in [self.eqm]) and len(self.eqm) == self.ns:
                break
        self.ns = len(self.eqm)
        # Turn everything into arrays, convert to kcal/mol
        self.eqm = np.array(self.eqm)
        self.eqm *= (eqcgmx / 4.184)

    def indicate(self):
        delta = (self.emm-self.eqm)
        deltanrm = self.prefactor*(delta/self.divisor)**2
        if len(self.label) == self.ns:
            PrintDict = OrderedDict()
            for i,label in enumerate(self.label):
                PrintDict[label] = "% 9.3f % 9.3f % 9.3f % 9.3f % 11.5f" % (self.emm[i], self.eqm[i], delta[i], self.divisor[i], deltanrm[i])
            printcool_dictionary(PrintDict,title="Target: %s\nInteraction Energies (kcal/mol), Objective = % .5e\n %-10s %9s %9s %9s %9s %11s" %
                                 (self.name, self.objective, "Label", "Calc.", "Ref.", "Delta", "Divisor", "Term"),keywidth=15)
        else:
            # logger.info("Target: %s Objective: % .5e (add LABEL keywords in qdata.txt for full printout)\n" % (self.name,self.objective))
            Headings = ["Observable", "Difference\nRMS (Calc-Ref)", "Denominator\n(Specified)", " Percent \nDifference"]
            Data = OrderedDict([])
            Data['Energy (kcal/mol)'] = ["%8.4f" % np.sqrt(np.mean(delta**2)),
                                       "%8.4f" % np.mean(self.divisor),
                                       "%.4f%%" % (np.sqrt(np.mean(delta/self.divisor)**2)*100)]
            self.printcool_table(data=Data, headings=Headings, color=0)
            logger.info("add LABEL keywords in qdata.txt to print out each snapshot\n")


        # if len(self.RMSDDict) > 0:x
        #     printcool_dictionary(self.RMSDDict,title="Geometry Optimized Systems (Angstrom), Objective = %.5e\n %-38s %11s %11s" % (self.rmsd_part, "System", "RMSD", "Term"), keywidth=45)

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}

        # If the weight is zero, turn all derivatives off.
        if (self.weight == 0.0):
            AGrad = False
            AHess = False

        def callM(mvals_, dielectric=False):
            logger.info("\r")
            pvals = self.FF.make(mvals_)
            return self.engine.interaction_energy(self.select1, self.select2)

        logger.info("Executing\r")
        emm = callM(mvals)

        D = emm - self.eqm
        dV = np.zeros((self.FF.np,len(emm)))

        if self.writelevel > 0:
            # Dump interaction energies to disk.
            np.savetxt('M.txt',emm)
            np.savetxt('Q.txt',self.eqm)
            import pickle
            pickle.dump((self.name, self.label, self.prefactor, self.eqm, emm), open("qm_vs_mm.p",'w'))
            # select the qm and mm data that has >0 weight to plot
            qm_data, mm_data = [], []
            for i in xrange(len(self.eqm)):
                if self.prefactor[i] != 0:
                    qm_data.append(self.eqm[i])
                    mm_data.append(emm[i])
            plot_interaction_qm_vs_mm(qm_data, mm_data, title="Interaction Energy "+self.name)

        # Do the finite difference derivative.
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = emm)
            # Create the force field one last time.
            pvals  = self.FF.make(mvals)

        Answer['X'] = np.dot(self.prefactor*D/self.divisor,D/self.divisor)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(self.prefactor*D/self.divisor, dV[p,:]/self.divisor)
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(self.prefactor*dV[p,:]/self.divisor, dV[q,:]/self.divisor)

        if not in_fd():
            self.emm = emm
            self.objective = Answer['X']

        ## QYD: try to clean up OpenMM engine.simulation objects to free up GPU memory
        try:
            if self.engine.name == 'openmm':
                if hasattr(self.engine, 'simulation'): del self.engine.simulation
                if hasattr(self.engine, 'A'): del self.engine.A
                if hasattr(self.engine, 'B'): del self.engine.B
        except:
            pass

        return Answer

def plot_interaction_qm_vs_mm(eqm, emm, title=''):
    import matplotlib.pyplot as plt
    plt.plot(eqm, label='QM Data', marker='^')
    plt.plot(emm, label='MM Data', marker='o')
    plt.legend()
    plt.xlabel('Snapshots')
    plt.ylabel('Interaction Energy (kcal/mol)')
    plt.title(title)
    plt.savefig("e_qm_vs_mm.pdf")
    plt.close()
