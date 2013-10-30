""" @package forcebalance.interaction Interaction energy fitting module.

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
import numpy as np
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, uncommadash, printcool_dictionary
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
        ## What is the energy denominator?
        self.set_option(tgt_opts,'energy_denom','energy_denom')
        ## Set fragment 1
        self.set_option(tgt_opts,'fragment1','fragment1')
        if len(self.fragment1) == 0:
            raise Exception('You need to define the first fragment using the fragment1 keyword')
        self.select1 = np.array(uncommadash(self.fragment1))
        ## Set fragment 2
        self.set_option(tgt_opts,'fragment2','fragment2')
        if len(self.fragment2) == 0:
            raise Exception('You need to define the second fragment using the fragment2 keyword')
        self.select2 = np.array(uncommadash(self.fragment2))
        ## Set upper cutoff energy
        self.set_option(tgt_opts,'energy_upper','energy_upper')
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## Reference (QM) interaction energies
        self.eqm           = []
        ## Snapshot label, useful for graphing
        self.label         = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.tgtdir,"qdata.txt")
        ## Qualitative Indicator: average energy error (in kJ/mol)
        self.e_err = 0.0
        self.e_err_pct = None
        ## Read in the trajectory file
        if self.ns == -1:
            self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords))
            self.ns = len(self.mol)
        else:
            self.mol = Molecule(os.path.join(self.root,self.tgtdir,self.coords))[:self.ns]
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,tgt_opts)

        logger.info("The energy denominator is: %s kcal/mol\n"  % str(self.energy_denom))
        # Internally things are handled in kJ/mol.
        denom = self.energy_denom * 4.184
        # Create the denominator.
        if self.cauchy:
            self.divisor = np.sqrt(self.eqm**2 + denom**2)
            if self.attenuate:
                raise Exception('attenuate and cauchy are mutually exclusive')
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
            logger.info("Interactions more repulsive than %s will not be fitted\n" % str(self.energy_upper))
            ecut = self.energy_upper * 4.184
            self.prefactor = 1.0 * (self.eqm < ecut)
        else:
            self.prefactor = np.ones(len(self.eqm))

    def read_reference_data(self):
        
        """ Read the reference ab initio data from a file such as qdata.txt.

        After reading in the information from qdata.txt, it is converted
        into the GROMACS/OpenMM units (kJ/mol for energy, kJ/mol/nm force).

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
        # Turn everything into arrays, convert to kJ/mol, and subtract the mean energy from the energy arrays
        self.eqm = np.array(self.eqm)
        self.eqm *= eqcgmx

    def prepare_temp_directory(self, options, tgt_opts):
        """ Prepare the temporary directory, by default does nothing """
        return
        
    def indicate(self):
        if len(self.label) == self.ns:
            PrintDict = OrderedDict()
            delta = (self.emm-self.eqm)
            deltanrm = self.prefactor*(delta/self.divisor)**2
            for i,label in enumerate(self.label):
                PrintDict[label] = "% 9.3f % 9.3f % 9.3f % 9.3f % 11.5f" % (self.emm[i]/4.184, self.eqm[i]/4.184, delta[i]/4.184, self.divisor[i]/4.184, deltanrm[i])
            printcool_dictionary(PrintDict,title="Target: %s\nInteraction Energies (kcal/mol), Objective = % .5e\n %-10s %9s %9s %9s %9s %11s" % 
                                 (self.name, self.objective, "Label", "Calc.", "Ref.", "Delta", "Divisor", "Term"),keywidth=15)
        else:
            logger.info("Target: %s Objective: % .5e (add LABEL keywords in qdata.txt for full printout)\n" % (self.name,self.objective))
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
            return self.interaction_driver_all(dielectric)

        logger.info("Executing\r")
        emm = callM(mvals)

        D = emm - self.eqm
        dV = np.zeros((self.FF.np,len(emm)))

        # Dump interaction energies to disk.
        np.savetxt('M.txt',emm)
        np.savetxt('Q.txt',self.eqm)

        # Do the finite difference derivative.
        if AGrad or AHess:
            for p in range(self.FF.np):
                dV[p,:], _ = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = emm)
            # Create the force field one last time.
            pvals  = self.FF.make(mvals)
                
        Answer['X'] = np.dot(self.prefactor*D/self.divisor,D/self.divisor)
        for p in range(self.FF.np):
            Answer['G'][p] = 2*np.dot(self.prefactor*D/self.divisor, dV[p,:]/self.divisor)
            for q in range(self.FF.np):
                Answer['H'][p,q] = 2*np.dot(self.prefactor*dV[p,:]/self.divisor, dV[q,:]/self.divisor)

        if not in_fd():
            self.emm = emm
            self.objective = Answer['X']

        return Answer

