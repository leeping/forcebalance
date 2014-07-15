""" @package forcebalance.vibration Vibrational mode fitting module

@author Lee-Ping Wang
@date 08/2012
"""

import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohrang, warn_press_key, pvec1d, pmat2d
import numpy as np
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from _assign import Assign
from collections import OrderedDict
#from _increment import Vibration_Build

from forcebalance.output import getLogger
logger = getLogger(__name__)

def count_assignment(assignment, verbose=True):
    for i in range(len(assignment)):
        if sum(assignment==i) != 1 and verbose:
            logger.info("Vibrational mode %i is assigned %i times\n" % (i+1, sum(assignment==i)))

class Vibration(Target):

    """ Subclass of Target for fitting force fields to vibrational spectra (from experiment or theory).

    Currently Tinker is supported.

    """
    
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Vibration,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts,'wavenumber_tol','denom')
        self.set_option(tgt_opts,'reassign_modes','reassign')
        self.set_option(tgt_opts,'normalize')
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The vdata.txt file that contains the vibrations.
        self.vfnm = os.path.join(self.tgtdir,"vdata.txt")
        ## Read in the reference data
        self.read_reference_data()
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(self.OptionDict.items() + options.items())
        del engine_args['name']
        ## Create engine object.
        self.engine = self.engine_(target=self, **engine_args)
        if self.FF.rigid_water:
            logger.error('This class cannot be used with rigid water molecules.\n')
            raise RuntimeError

    def read_reference_data(self):
        """ Read the reference vibrational data from a file. """
        ## Number of atoms
        self.na = -1
        self.ref_eigvals = []
        self.ref_eigvecs = []
        an = 0
        ln = 0
        cn = -1
        for line in open(self.vfnm):
            line = line.split('#')[0] # Strip off comments
            s = line.split()
            if len(s) == 1 and self.na == -1:
                self.na = int(s[0])
                xyz = np.zeros((self.na, 3))
                cn = ln + 1
            elif ln == cn:
                pass
            elif an < self.na and len(s) == 4:
                xyz[an, :] = np.array([float(i) for i in s[1:]])
                an += 1
            elif len(s) == 1:
                if float(s[0]) < 0:
                    logger.warning('Warning: Setting imaginary frequency = % .3fi to zero.\n' % abs(float(s[0])))
                    self.ref_eigvals.append(0.0)
                else:
                    self.ref_eigvals.append(float(s[0]))
                self.ref_eigvecs.append(np.zeros((self.na, 3)))
                an = 0
            elif len(s) == 3:
                self.ref_eigvecs[-1][an, :] = np.array([float(i) for i in s])
                an += 1
            elif len(s) == 0:
                pass
            else:
                logger.info(line + '\n')
                logger.error("This line doesn't comply with our vibration file format!\n")
                raise RuntimeError
            ln += 1
        self.ref_eigvals = np.array(self.ref_eigvals)
        self.ref_eigvecs = np.array(self.ref_eigvecs)
        for v2 in self.ref_eigvecs:
            v2 /= np.linalg.norm(v2)
        return

    def indicate(self):
        """ Print qualitative indicator. """
        if self.reassign == 'overlap' : count_assignment(self.c2r)
        banner = "Frequencies (wavenumbers)"
        headings = ["Mode #", "Reference", "Calculated", "Difference", "Ref(dot)Calc"]
        data = OrderedDict([(i, ["%.4f" % self.ref_eigvals[i], "%.4f" % self.calc_eigvals[i], "%.4f" % (self.calc_eigvals[i] - self.ref_eigvals[i]), "%.4f" % self.overlaps[i]]) for i in range(len(self.ref_eigvals))])
        self.printcool_table(data, headings, banner)
        return

    def vibration_driver(self):
        if hasattr(self, 'engine') and hasattr(self.engine, 'normal_modes'):
            return self.engine.normal_modes()
        else:
            logger.error('Normal mode calculation not supported, try using a different engine\n')
            raise NotImplementedError


    def process_vectors(self, vecs, verbose=False, check=False):
        """ Return a set of normal and mass-weighted eigenvectors such that their outer product is the identity. """
        mw = np.array(self.engine.AtomLists['Mass'])
        vecs_mw = vecs.copy()
        for i in range(len(mw)):
            vecs_mw[:, i, :] *= mw[i]
        rnrms = np.array([np.dot(v1.flatten(), v2.flatten()) for v1, v2 in zip(vecs, vecs_mw)])
        vecs_nrm = vecs.copy()
        for i in range(len(rnrms)):
            vecs_nrm[i, :, :] /= np.sqrt(rnrms[i])
        vecs_nrm_mw = vecs_nrm.copy()
        for i in range(len(mw)):
            vecs_nrm_mw[:, i, :] *= mw[i]
        if verbose:
            pmat2d(np.array([[np.dot(v1.flatten(), v2.flatten()) for v1 in vecs_nrm] for v2 in vecs_nrm_mw]), precision=3, format="f")
            logger.info('\n')
        if check:
            for i, v1 in enumerate(vecs_nrm):
                for j, v2 in enumerate(vecs_nrm_mw):
                    if np.abs(np.dot(v1.flatten(), v2.flatten()) - float(i==j)) > 1e-1:
                        logger.warn("In modes %i %i, orthonormality violated by %f\n" % (i, j, np.dot(v1.flatten(), v2.flatten()) - float(i==j)))
        return vecs_nrm, vecs_nrm_mw

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}

        if not hasattr(self, 'ref_eigvecs_nrm'):
            self.ref_eigvecs_nrm, self.ref_eigvecs_nrm_mw = self.process_vectors(self.ref_eigvecs)
        
        def get_eigvals(mvals_):
            self.FF.make(mvals_)
            eigvals, eigvecs = self.vibration_driver()
            eigvecs_nrm, eigvecs_nrm_mw = self.process_vectors(eigvecs)
            # The overlap metric may take into account some frequency differences
            dev = np.array([[(np.abs(i-j)/1000)/(1.0+np.abs(i-j)/1000) for j in self.ref_eigvals] for i in eigvals])
            for i in range(dev.shape[0]):
                dev[i, :] /= max(dev[i, :])

            if self.reassign in ['permute', 'overlap']:
                # In the matrix that we constructed, these are the column numbers (reference mode numbers) 
                # that are mapped to the row numbers (calculated mode numbers)
                if self.reassign == 'permute':
                    a = np.array([[int(1e6*(1.0-np.dot(v1.flatten(),v2.flatten())**2)) for v2 in self.ref_eigvecs_nrm] for v1 in eigvecs_nrm_mw])
                    c2r = Assign(a)
                    eigvals = eigvals[c2r]
                elif self.reassign == 'overlap':
                    a = np.array([[(1.0-np.dot(v1.flatten(),v2.flatten())**2) for v2 in self.ref_eigvecs_nrm] for v1 in eigvecs_nrm_mw])
                    a += dev
                    c2r = np.argmin(a, axis=0)
                    eigvals_p = []
                    for j in c2r:
                        eigvals_p.append(eigvals[j])
                    eigvals = np.array(eigvals_p)
            if not in_fd():
                if self.reassign == 'permute':
                    eigvecs_nrm_mw = eigvecs_nrm_mw[c2r]
                elif self.reassign == 'overlap':
                    self.c2r = c2r
                    eigvecs_nrm_mw_p = []
                    for j in c2r:
                        eigvecs_nrm_mw_p.append(eigvecs_nrm_mw[j])
                    eigvecs_nrm_mw = np.array(eigvecs_nrm_mw_p)
                self.overlaps = np.array([np.abs(np.dot(v1.flatten(),v2.flatten())) for v1, v2 in zip(self.ref_eigvecs_nrm, eigvecs_nrm_mw)])
            return eigvals

        calc_eigvals = get_eigvals(mvals)
        D = calc_eigvals - self.ref_eigvals
        dV = np.zeros((self.FF.np,len(calc_eigvals)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p,:], _ = f12d3p(fdwrap(get_eigvals, mvals, p), h = self.h, f0 = calc_eigvals)
        Answer['X'] = np.dot(D,D) / self.denom**2 / (len(D) if self.normalize else 1)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(D, dV[p,:]) / self.denom**2 / (len(D) if self.normalize else 1)
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dV[p,:], dV[q,:]) / self.denom**2 / (len(D) if self.normalize else 1)
        if not in_fd():
            self.calc_eigvals = calc_eigvals
            self.objective = Answer['X']
        return Answer
