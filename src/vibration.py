""" @package forcebalance.vibration Vibrational mode fitting module

@author Lee-Ping Wang
@date 08/2012
"""
from __future__ import division

from builtins import zip
from builtins import range
import os
import shutil
from forcebalance.nifty import col, eqcgmx, flat, floatornan, fqcgmx, invert_svd, kb, printcool, bohr2ang, warn_press_key, pvec1d, pmat2d
import numpy as np
from forcebalance.target import Target
from forcebalance.molecule import Molecule, format_xyz_coord
from re import match, sub
import subprocess
from subprocess import PIPE
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
# from ._assign import Assign
from scipy import optimize
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
        ## LPW 2018-02-11: This is set to True if the target calculates
        ## a single-point property over several existing snapshots.
        self.loop_over_snapshots = False
        ## The vdata.txt file that contains the vibrations.
        self.vfnm = os.path.join(self.tgtdir,"vdata.txt")
        ## Read in the reference data
        self.na, _, self.ref_eigvals, self.ref_eigvecs = read_reference_vdata(self.vfnm)
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(list(self.OptionDict.items()) + list(options.items()))
        engine_args.pop('name', None)
        ## Create engine object.
        self.engine = self.engine_(target=self, **engine_args)
        if self.FF.rigid_water:
            logger.error('This class cannot be used with rigid water molecules.\n')
            raise RuntimeError

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

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}

        def get_eigvals(mvals_):
            self.FF.make(mvals_)
            eigvals, eigvecs = self.vibration_driver()
            # The overlap metric may take into account some frequency differences.
            # Here, an element of dev is equal to 2/3 if (for example) the frequencies differ by 1000.
            dev = np.array([[(np.abs(i-j)/1000)/(1.0+np.abs(i-j)/1000) for j in eigvals] for i in self.ref_eigvals])
            for i in range(dev.shape[0]):
                dev[i, :] /= max(dev[i, :])

            if self.reassign in ['permute', 'overlap']:
                # The elements of "a" matrix are the column numbers (reference mode numbers) 
                # that are mapped to the row numbers (calculated mode numbers).
                # Highly similar eigenvectors are assigned small values because
                # the assignment problem is a cost minimization problem.
                a = np.array([[(1.0-vib_overlap(self.engine, v1, v2)) for v2 in eigvecs] for v1 in self.ref_eigvecs])
                a += dev
                if self.reassign == 'permute':
                    row, c2r = optimize.linear_sum_assignment(a)
                    eigvals = eigvals[c2r]
                elif self.reassign == 'overlap':
                    c2r = np.argmin(a, axis=0)
                    eigvals_p = []
                    for j in c2r:
                        eigvals_p.append(eigvals[j])
                    eigvals = np.array(eigvals_p)
            if not in_fd():
                if self.reassign == 'permute':
                    eigvecs = eigvecs[c2r]
                elif self.reassign == 'overlap':
                    self.c2r = c2r
                    eigvecs_p = []
                    for j in c2r:
                        eigvecs_p.append(eigvecs[j])
                    eigvecs = np.array(eigvecs_p)
                self.overlaps = np.array([vib_overlap(self.engine, v1, v2) for v1, v2 in zip(self.ref_eigvecs, eigvecs)])
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

def vib_overlap(engine, v1, v2):
    """
    Calculate overlap between vibrational modes for two Cartesian displacements.

    Parameters
    ----------
    v1, v2 : np.ndarray
        The two sets of Cartesian displacements to compute overlap for,
        3*N_atoms values each.

    Returns
    -------
    float
        Overlap between mass-weighted eigenvectors
    """
    if hasattr(engine, 'realAtomIdxs'): realAtoms = engine.realAtomIdxs
    else: realAtoms = [i for i in range(len(engine.AtomLists['Mass']))]
    sqrtm = np.sqrt(np.array(engine.AtomLists['Mass'])[realAtoms])
    v1m = v1.copy()
    v1m *= sqrtm[:, np.newaxis]
    v1m /= np.linalg.norm(v1m)
    v2m = v2.copy()
    v2m *= sqrtm[:, np.newaxis]
    v2m /= np.linalg.norm(v2m)
    return np.abs(np.dot(v1m.flatten(), v2m.flatten()))

def read_reference_vdata(vfnm):
    """ Read the reference vibrational data from a file. """
    ## Number of atoms
    na = -1
    ref_eigvals = []
    ref_eigvecs = []
    an = 0
    ln = 0
    cn = -1
    for line in open(vfnm):
        line = line.split('#')[0] # Strip off comments
        s = line.split()
        if len(s) == 1 and na == -1:
            na = int(s[0])
            xyz = np.zeros((na, 3))
            cn = ln + 1
        elif ln == cn:
            pass
        elif an < na and len(s) == 4:
            xyz[an, :] = np.array([float(i) for i in s[1:]])
            an += 1
        elif len(s) == 1:
            if float(s[0]) < 0:
                logger.warning('Warning: Setting imaginary frequency = % .3fi to zero.\n' % abs(float(s[0])))
                ref_eigvals.append(0.0)
            else:
                ref_eigvals.append(float(s[0]))
            ref_eigvecs.append(np.zeros((na, 3)))
            an = 0
        elif len(s) == 3:
            ref_eigvecs[-1][an, :] = np.array([float(i) for i in s])
            an += 1
        elif len(s) == 0:
            pass
        else:
            logger.info(line + '\n')
            logger.error("This line doesn't comply with our vibration file format!\n")
            raise RuntimeError
        ln += 1
    ref_eigvals = np.array(ref_eigvals)
    ref_eigvecs = np.array(ref_eigvecs)
    for v2 in ref_eigvecs:
        v2 /= np.linalg.norm(v2)
    return na, xyz, ref_eigvals, ref_eigvecs
