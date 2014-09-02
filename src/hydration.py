""" @package forcebalance.hydration Hydration free energy fitting module

@author Lee-Ping Wang
@date 09/2014
"""

import os
import shutil
import numpy as np
from forcebalance.target import Target
from forcebalance.molecule import Molecule
from re import match, sub
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import OrderedDict

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Hydration(Target):

    """ Subclass of Target for fitting force fields to hydration free energies."""
    
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Hydration,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts,'hfedata_txt','datafile')
        self.set_option(tgt_opts,'normalize')
        self.set_option(tgt_opts,'optimize_geometry')
        self.set_option(tgt_opts,'energy_denom','denom')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The vdata.txt file that contains the hydrations.
        self.datafile = os.path.join(self.tgtdir,self.datafile)
        ## Read in the reference data
        self.read_reference_data()
        ## Copy target options into engine kwargs.
        self.engine_args = OrderedDict(self.OptionDict.items() + options.items())
        del self.engine_args['name']
        ## Create engine objects.
        self.build_engines()
        if self.FF.rigid_water:
            logger.error('This class cannot be used with rigid water molecules.\n')
            raise RuntimeError

    def read_reference_data(self):
        """ Read the reference hydrational data from a file. """
        self.refdata = OrderedDict([(l.split()[0], float(l.split()[1])) for l in open(self.datafile).readlines()])

    def build_engines(self):
        self.engines = OrderedDict()
        self.aq_engines = OrderedDict()
        self.gas_engines = OrderedDict()
        for mnm in self.refdata.keys():
            pdbfnm = os.path.abspath(os.path.join(self.root,self.tgtdir, 'molecules', mnm+'.pdb'))
            self.aq_engines[mnm] = self.engine_(target=self, coords=pdbfnm, implicit_solvent=True, **self.engine_args)
            self.gas_engines[mnm] = self.engine_(target=self, coords=pdbfnm, implicit_solvent=False, **self.engine_args)

    def indicate(self):
        """ Print qualitative indicator. """
        banner = "Hydration free energies (kcal/mol)"
        headings = ["Molecule", "Reference", "Calculated", "Difference", "Residual"]
        data = OrderedDict([(i, ["%.4f" % self.refdata[i], "%.4f" % self.calc[i], "%.4f" % (self.calc[i] - self.refdata[i]), 
                                 "%.4f" % (self.calc[i] - self.refdata[i])**2]) for i in self.refdata.keys()])
        self.printcool_table(data, headings, banner)

    def hydration_driver(self):
        hfe = OrderedDict()
        for mnm in self.refdata.keys():
            eaq, rmsdaq = self.aq_engines[mnm].optimize()
            egas, rmsdgas = self.gas_engines[mnm].optimize()
            hfe[mnm] = eaq - egas
        return hfe

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}

        def get_hfe(mvals_):
            self.FF.make(mvals_)
            self.hfe_dict = self.hydration_driver()
            return np.array(self.hfe_dict.values())

        calc_hfe = get_hfe(mvals)
        D = calc_hfe - np.array(self.refdata.values())
        dD = np.zeros((self.FF.np,len(self.refdata.keys())))
        if AGrad or AHess:
            for p in self.pgrad:
                dD[p,:], _ = f12d3p(fdwrap(get_hfe, mvals, p), h = self.h, f0 = calc_hfe)
        Answer['X'] = np.dot(D,D) / self.denom**2 / (len(D) if self.normalize else 1)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(D, dD[p,:]) / self.denom**2 / (len(D) if self.normalize else 1)
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dD[p,:], dD[q,:]) / self.denom**2 / (len(D) if self.normalize else 1)
        if not in_fd():
            self.calc = self.hfe_dict
            self.objective = Answer['X']
        return Answer
