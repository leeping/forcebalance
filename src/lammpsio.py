import os, sys, re
import copy
from re import match, sub, split, findall
import networkx as nx
from forcebalance.nifty import isint, isfloat, _exec, CopyFile, LinkFile, warn_once, which, onefile, listfiles, warn_press_key, wopen
import numpy as np
from forcebalance import BaseReader
from forcebalance.engine import Engine
from forcebalance.abinitio import AbInitio
from forcebalance.molecule import Molecule
from collections import OrderedDict, defaultdict

from forcebalance.output import getLogger
logger = getLogger(__name__)

# LAMMPS SPECIFIC IMPORTS
import ctypes as ct
import lammps, shutil, time
from pprint import pprint
from forcebalance.optimizer import Counter

class LAMMPS_INTERFACE(object):

    def __init__(self, fn_lmp):
        # Load all LAMMPS commands
        lammps_in = open(fn_lmp).read().splitlines()
        self._lammps_in = lammps_in

        # Neighbor lists checked at every step
        lammps_in.append('neigh_modify delay 0 every 1 check yes')

        # Instantiate
        # lmp = lammps.lammps()
        lmp = lammps.lammps(cmdargs=['-screen','none','-log','none'])
        self._lmp = lmp

        # Command execution
        for cmd in self._lammps_in:
            lmp.command(cmd)

        # Gather some constants
        natoms = lmp.get_natoms()
        self.natoms = natoms
        N = 3 * natoms
        self.N = N

        # Ctypes array for LAMMPS position data
        self._x_c = (N * ct.c_double)()
        self._dV_dx = (N * ct.c_double)()

    def get_V(self, x, force=False):
        # Flatten array if necessary
        if len(x.shape) != 1:
            x_ = x.flatten()
        else:
            x_ = x
            
        # Our LAMMPS instance
        lmp = self._lmp

        # Update LAMMPS positions 
        self._x_c[:] = x_
        lmp.scatter_atoms('x', 2, 3, self._x_c)

        # Update energy and forces in LAMMPS
        lmp.command('run 0 pre yes post no')

        # Get energy and forces from LAMMPS
        V = float(lmp.extract_compute('thermo_pe', 0, 0))

        if force:
            F_c = lmp.gather_atoms('f', 1, 3)
            return V, F_c[:]
        return V

class LAMMPS(Engine):
    """ Engine for carrying out LAMMPS calculations. """
    
    def __init__(self, name="lammps", **kwargs):
        ## Keyword args that aren't in this list are filtered out.
        self.valkwd = ['lammps_in']
        super(LAMMPS,self).__init__(name=name, **kwargs)

    def setopts(self, **kwargs):
        """ Called by __init__ ; Set LAMMPS-specific options and load input file. """
        if 'lammps_in' in kwargs:
            self.lammps_in = kwargs['lammps_in']
        elif hasattr(self, 'target'):
            self.lammps_in = self.target.lammps_in
        else:
            raise RuntimeError('Need to set lammps_in')

    def readsrc(self, **kwargs):
        """ Called by __init__ ; read files from the source directory. """

        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs:
            self.mol = Molecule(kwargs['coords'])
        else:
            logger.error('Must provide either a molecule object or coordinate file.\n')
            raise RuntimeError
        # if 'mol' in kwargs:
        #     self.mol = kwargs['mol']
        # else:
        #     raise RuntimeError('Coordinate file missing from target \
        #                         directory.')
        
    def prepare(self, pbc=False, **kwargs):
        """ Called by __init__ ; prepare the temp directory. """
        ## These are required by abinitio.py
        self.AtomLists = defaultdict(list)
        self.AtomMask  = [1 for i in range(self.mol.na)]
        ## Extract positions from molecule object
        self.xyz_snapshots = []
        for I in range(len(self.mol)):
            self.xyz_snapshots.append(self.mol.xyzs[I])
        ## Assume LAMMPS atom types are assigned in the same order in which elements appear
        reaxff_atom_types = self.mol.get_reaxff_atom_types()
        ## Write first frame as LAMMPS data file
        self.mol[0].write(os.path.join(self.tempdir, "lammps.data"), ftype="lammps")
        ## Search for LAMMPS input file
        if hasattr(self, 'target') and os.path.exists(os.path.join(self.target.tgtdir, self.lammps_in)):
            abs_lammps_in = os.path.join(self.target.tgtdir, self.lammps_in)
        elif os.path.exists(os.path.join(os.path.split(__file__)[0],"data",self.lammps_in)):
            abs_lammps_in = os.path.join(os.path.split(__file__)[0],"data",self.lammps_in)
        else:
            raise RuntimeError("LAMMPS input file %s not found. Make sure it is in the target folder, or in the ForceBalance src/data folder.")
        ## Write LAMMPS input file, making appropriate substitutions
        with open(os.path.join(self.tempdir, self.lammps_in), "w") as f:
            for line in open(os.path.join(os.path.split(__file__)[0],"data","reaxff.in")):
                print >> f, line.replace("REPLACE_WITH_FFIELD",self.FF.fnms[0]).replace("REPLACE_WITH_ATOMS"," ".join(reaxff_atom_types)),
        ## Copy over ReaxFF control file
        CopyFile(os.path.join(os.path.split(__file__)[0],"data","reaxff.control"),os.path.join(self.tempdir,"reaxff.control"))
        ## Create interface to LAMMPS
        ## self.create_interfaces()
   
    def create_interfaces(self):
        """
        Create interfaces to LAMMPS.  This used to contain several interfaces.  Right now it is vestigial.
        """
        if not os.path.exists(self.FF.fnms[0]): self.FF.make()
        self._lmp_main = LAMMPS_INTERFACE(self.lammps_in)
        # root = os.getcwd()
        # tempdir = self.tempdir
        # os.chdir(os.path.join(tempdir))
        # self.FF.make()
        # self._lmp_main = LAMMPS_INTERFACE("reaxff.in")
        # os.chdir(root)

    def evaluate_(self, force=False):
        """ 
        Utility function for computing energy and forces using LAMMPS
        """
        # Must update the interface in order to re-read the force field
        self.create_interfaces()
        Result = {}
        snaps = self.xyz_snapshots
        energies = []
        forces = []
        for coords in snaps:
            if force:
                E, F = self._lmp_main.get_V(coords, force=True)
                F = np.array(F)
            else:
                E = self._lmp_main.get_V(coords)
            energies.append(E * 4.184)
            if force:
                forces.append(F * 4.184 * 10)
        Result['Energy'] = np.array(energies)
        if force: Result['Force'] = np.array(forces)
        del self._lmp_main
        return Result

    def energy_force(self):
        """ Loop through the snapshots and compute the energies and forces using LAMMPS. """
        Result = self.evaluate_(force=True)
        E = Result["Energy"].reshape(-1,1)
        F = Result["Force"]
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy(self):
        return self.evaluate_()["Energy"]
    
class AbInitio_LAMMPS(AbInitio):

    """Subclass of Target for force and energy matching
    using LAMMPS."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Coordinate file.
        self.set_option(tgt_opts,'coords',default='all.xyz')
        ## LAMMPS input file.
        self.set_option(tgt_opts, 'lammps_in', default='reaxff.in')
        ## Name of the engine.
        self.engine_ = LAMMPS
        ## Initialize base class.
        super(AbInitio_LAMMPS,self).__init__(options,tgt_opts,forcefield)

