""" @package interaction Interaction energy fitting module.

@author Lee-Ping Wang
@date 05/2012
"""

import os
import shutil
from nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, uncommadash, printcool_dictionary
from numpy import append, array, diag, dot, exp, log, mat, mean, ones, outer, sqrt, where, zeros, linalg, savetxt
from target import Target
from molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict

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
        ## What is the energy denominator?
        self.set_option(tgt_opts,'energy_denom','energy_denom')
        ## Set fragment 1
        self.set_option(tgt_opts,'fragment1','fragment1')
        if len(self.fragment1) == 0:
            raise Exception('You need to define the first fragment using the fragment1 keyword')
        self.select1 = array(uncommadash(self.fragment1))
        ## Set fragment 2
        self.set_option(tgt_opts,'fragment2','fragment2')
        if len(self.fragment2) == 0:
            raise Exception('You need to define the second fragment using the fragment2 keyword')
        self.select2 = array(uncommadash(self.fragment2))
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
            self.traj = Molecule(os.path.join(self.root,self.tgtdir,self.trajfnm))
            self.ns = len(self.traj)
        else:
            self.traj = Molecule(os.path.join(self.root,self.tgtdir,self.trajfnm))[:self.ns]
        ## Read in the reference data
        self.read_reference_data()
        ## Prepare the temporary directory
        self.prepare_temp_directory(options,tgt_opts)

        print "The energy denominator is:", self.energy_denom, "kcal/mol"
        # Internally things are handled in kJ/mol.
        denom = self.energy_denom * 4.184
        # Create the denominator.
        if self.cauchy:
            self.divisor = sqrt(self.eqm**2 + denom**2)
        else:
            self.divisor = ones(len(self.eqm)) * denom
        if self.cauchy:
            print "Each contribution to the interaction energy objective function will be scaled by 1.0 / ( energy_denom**2 + reference**2 )"
        if self.energy_upper > 0:
            print "Interactions more repulsive than", self.energy_upper, "will not be fitted"
            ecut = self.energy_upper * 4.184
            self.prefactor = 1.0 * (self.eqm < ecut)
        else:
            self.prefactor = ones(len(self.eqm))

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
        self.eqm = array(self.eqm)
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
            print "Target: %s Objective: % .5e (add LABEL keywords in qdata.txt for full printout)" % (self.name,self.objective)
        # if len(self.RMSDDict) > 0:
        #     printcool_dictionary(self.RMSDDict,title="Geometry Optimized Systems (Angstrom), Objective = %.5e\n %-38s %11s %11s" % (self.rmsd_part, "System", "RMSD", "Term"), keywidth=45)

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':zeros(self.FF.np, dtype=float), 'H':zeros((self.FF.np, self.FF.np), dtype=float)}
        
        # If the weight is zero, turn all derivatives off.
        if (self.weight == 0.0):
            AGrad = False
            AHess = False

        def callM(mvals_, dielectric=False):
            print "\r",
            pvals = self.FF.make(mvals_, self.usepvals)
            return self.interaction_driver_all(dielectric)

        print "Executing\r",
        emm = callM(mvals)

        D = emm - self.eqm
        dV = zeros((self.FF.np,len(emm)),dtype=float)

        # Dump interaction energies to disk.
        savetxt('M.txt',emm)
        savetxt('Q.txt',self.eqm)

        # Do the finite difference derivative.
        if AGrad or AHess:
            for p in range(self.FF.np):
                dV[p,:], _ = f12d3p(fdwrap(callM, mvals, p), h = self.h, f0 = emm)
            # Create the force field one last time.
            pvals  = self.FF.make(mvals,self.usepvals)
                
        Answer['X'] = dot(self.prefactor*D/self.divisor,D/self.divisor)
        for p in range(self.FF.np):
            Answer['G'][p] = 2*dot(self.prefactor*D/self.divisor, dV[p,:]/self.divisor)
            for q in range(self.FF.np):
                Answer['H'][p,q] = 2*dot(self.prefactor*dV[p,:]/self.divisor, dV[q,:]/self.divisor)

        if not in_fd():
            self.emm = emm
            self.objective = Answer['X']

        return Answer

