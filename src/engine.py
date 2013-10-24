""" Engine base class from which all ForceBalance MD engines are derived. """

import abc
import os
import subprocess
import shutil
import numpy as np
import time
from collections import OrderedDict
import tarfile
import forcebalance
from forcebalance.nifty import row, col, printcool_dictionary, link_dir_contents, createWorkQueue, getWorkQueue, wq_wait1, getWQIds, warn_once
from forcebalance.finite_difference import fdwrap_G, fdwrap_H, f1d2p, f12d3p
from forcebalance.optimizer import Counter
from forcebalance.output import getLogger
logger = getLogger(__name__)

class Engine(forcebalance.BaseClass):

    """
    Base class for all engines.

    1. Introduction
    
    In ForceBalance an Engine represents a molecular dynamics code
    and the calculations that may be carried out with that code.

    The Engine implements methods that execute operations such as:
    - Input: trajectory object
    -  Return: energy over trajectory
    -  Return: energy and force over trajectory
    -  Return: electrostatic potential over trajectory
    - How about:
    -  evaluate_snapshots(Molecule, Energy=True, Force=True, ESP=True)
       where all information passed in belongs in the Molecule object. :)
    -  Return a dictionary

    - Input: molecular geometry, 
    -  Return: optimized geometry
    -  Return: vibrational modes at optimized geometry
    -  Return: multipole moments at optimized geometry
    -  evaluate_optimized(Molecule, Energy=True, Frequencies=True, Moments=True)

    - Engine objects may be initialized using a molecule object and a
      force field object.

    2. Purpose

    Previously system calls to MD software have been made by the
    Target.  Duplication of code was occurring, because different
    Targets were carrying out the same type of calculation.

    3. Also
    
    Target objects should contain Engine objects, because OpenMM
    Engine objects need to be initialized at the start of a
    calculation.

    """

    def __init__(self, name="engine", **kwargs):
        super(Engine, self).__init__(kwargs)
        self.name = name
        if 'target' in kwargs:
            self.target = kwargs['target']
            self.root = self.target.root
        else:
            warn_once("Running without a target, using current directory.")
            self.root = os.getcwd()
        if hasattr(self,'target'):
            self.srcdir = os.path.join(self.root, self.target.tgtdir)
        else:
            self.srcdir = self.root
        if 'verbose' in kwargs:
            self.verbose = verbose
        else:
            self.verbose = False
        return

    def prepare(self):
        return

    def postinit(self):
        """ Perform post-initialization tasks. """
        if self.verbose:
            printcool_dictionary(OrderedDict([(i, self.__dict__[i]) for i in sorted(self.__dict__.keys())]), title="Attributes for engine %s" % self.__class__.__name__)
        self.prepare()

    @abc.abstractmethod
    def evaluate_snapshots(self, M):
        """ Evaluate properties over a collection of snapshots. """
        raise NotImplementedError('This method is not implemented in the base class')

    @abc.abstractmethod
    def evaluate_optimized(self, M):
        """ Evaluate properties on the optimized geometry. """
        raise NotImplementedError('This method is not implemented in the base class')

