""" @package forcebalance.moments Multipole moment fitting module

@author Lee-Ping Wang
@date 09/2012
"""

import os
import shutil
import numpy as np
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, printcool_dictionary, bohrang, warn_press_key
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
import itertools
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Moments(Target):

    """ Subclass of Target for fitting force fields to multipole moments (from experiment or theory).

    Currently Tinker is supported.

    """
    
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Moments,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts, 'dipole_denom')
        self.set_option(tgt_opts, 'quadrupole_denom')
        self.set_option(tgt_opts, 'polarizability_denom')
        self.set_option(tgt_opts, 'optimize_geometry')

	self.denoms = {}
        self.denoms['dipole'] = self.dipole_denom
        self.denoms['quadrupole'] = self.quadrupole_denom
        self.denoms['polarizability'] = self.polarizability_denom
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The mdata.txt file that contains the moments.
        self.mfnm = os.path.join(self.tgtdir,"mdata.txt")
        ## Dictionary of reference multipole moments.
        self.ref_moments = OrderedDict()
        ## Read in the reference data
        self.read_reference_data()
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(self.OptionDict.items() + options.items())
        del engine_args['name']
        ## Create engine object.
        self.engine = self.engine_(target=self, **engine_args)

    def read_reference_data(self):
        """ Read the reference data from a file. """
        ## Number of atoms
        self.na = -1
        self.ref_eigvals = []
        self.ref_eigvecs = []
        an = 0
        ln = 0
        cn = -1
        dn = -1
        qn = -1
        pn = -1
        for line in open(self.mfnm):
            line = line.split('#')[0] # Strip off comments
            s = line.split()
            if len(s) == 0:
                pass
            elif len(s) == 1 and self.na == -1:
                self.na = int(s[0])
                xyz = np.zeros((self.na, 3))
                cn = ln + 1
            elif ln == cn:
                pass
            elif an < self.na and len(s) == 4:
                xyz[an, :] = np.array([float(i) for i in s[1:]])
                an += 1
            elif an == self.na and s[0].lower() == 'dipole':
                dn = ln + 1
            elif ln == dn:
                self.ref_moments['dipole'] = OrderedDict(zip(['x','y','z'],[float(i) for i in s]))
            elif an == self.na and s[0].lower() in ['quadrupole', 'quadrapole']:
                qn = ln + 1
            elif ln == qn:
                self.ref_moments['quadrupole'] = OrderedDict([('xx',float(s[0]))])
            elif qn > 0 and ln == qn + 1:
                self.ref_moments['quadrupole']['xy'] = float(s[0])
                self.ref_moments['quadrupole']['yy'] = float(s[1])
            elif qn > 0 and ln == qn + 2:
                self.ref_moments['quadrupole']['xz'] = float(s[0])
                self.ref_moments['quadrupole']['yz'] = float(s[1])
                self.ref_moments['quadrupole']['zz'] = float(s[2])
            elif an == self.na and s[0].lower() in ['polarizability', 'alpha']:
                pn = ln + 1
            elif ln == pn:
                self.ref_moments['polarizability'] = OrderedDict([('xx',float(s[0]))])
                self.ref_moments['polarizability']['yx'] = float(s[1])
                self.ref_moments['polarizability']['zx'] = float(s[2])
            elif pn > 0 and ln == pn + 1:
                self.ref_moments['polarizability']['xy'] = float(s[0])
                self.ref_moments['polarizability']['yy'] = float(s[1])
                self.ref_moments['polarizability']['zy'] = float(s[2])
            elif pn > 0 and ln == pn + 2:
                self.ref_moments['polarizability']['xz'] = float(s[0])
                self.ref_moments['polarizability']['yz'] = float(s[1])
                self.ref_moments['polarizability']['zz'] = float(s[2])
            else:
                logger.info("%s\n" % line)
                logger.error("This line doesn't comply with our multipole file format!\n")
                raise RuntimeError
            ln += 1
        # Subtract the trace of the quadrupole moment.
        if 'quadrupole' in self.ref_moments:
            trace3 = (self.ref_moments['quadrupole']['xx'] + self.ref_moments['quadrupole']['yy'] + self.ref_moments['quadrupole']['zz'])/3
            self.ref_moments['quadrupole']['xx'] -= trace3
            self.ref_moments['quadrupole']['yy'] -= trace3
            self.ref_moments['quadrupole']['zz'] -= trace3

        return

    def indicate(self):
        """ Print qualitative indicator. """
        logger.info("\rTarget: %-15s\n" % self.name)
        #print "Multipole Moments and Po"
        #print "Reference :", self.ref_moments
        #print "Calculated:", self.calc_moments
        #print "Objective = %.5e" % self.objective

        ref_momvals = self.unpack_moments(self.ref_moments)
        calc_momvals = self.unpack_moments(self.calc_moments)
        PrintDict = OrderedDict()
        i = 0
        for Ord in self.ref_moments:
            for Comp in self.ref_moments[Ord]:
                if abs(self.calc_moments[Ord][Comp]) > 1e-6 or abs(self.ref_moments[Ord][Comp]) > 1e-6:
                    PrintDict["%s-%s" % (Ord, Comp)] = "% 9.3f % 9.3f % 9.3f % 12.5f" % (self.calc_moments[Ord][Comp],
                                                                                         self.ref_moments[Ord][Comp],
                                                                                         self.calc_moments[Ord][Comp]-self.ref_moments[Ord][Comp],
                                                                                         (ref_momvals[i] - calc_momvals[i])**2)

                i += 1
                
        printcool_dictionary(PrintDict,title="Moments and/or Polarizabilities, Objective = % .5e\n %-20s %9s %9s %9s %11s" % 
                             (self.objective, "Component", "Calc.", "Ref.", "Delta", "Term"))

        return

    def unpack_moments(self, moment_dict):
        answer = np.array(list(itertools.chain(*[[dct[i]/self.denoms[ord] if self.denoms[ord] != 0.0 else 0.0 for i in dct] for ord,dct in moment_dict.items()])))
        return answer

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        def get_momvals(mvals_):
            self.FF.make(mvals_)
            moments = self.engine.multipole_moments(polarizability='polarizability' in self.ref_moments, optimize=self.optimize_geometry)
            # Unpack from dictionary.
            return self.unpack_moments(moments)

        self.FF.make(mvals)
        ref_momvals = self.unpack_moments(self.ref_moments)
        calc_moments = self.engine.multipole_moments(polarizability='polarizability' in self.ref_moments, optimize=self.optimize_geometry)
        calc_momvals = self.unpack_moments(calc_moments)

        D = calc_momvals - ref_momvals
        dV = np.zeros((self.FF.np,len(calc_momvals)))

        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(get_momvals, mvals, p), h = self.h, f0 = calc_momvals)
                
        Answer['X'] = np.dot(D,D)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(D, dV[p,:])
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:])

        if not in_fd():
            self.calc_moments = calc_moments
            self.objective = Answer['X']

        return Answer
