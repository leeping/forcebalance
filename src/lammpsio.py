import os, sys, re
import copy
from re import match, sub, split, findall
import networkx as nx
from forcebalance.nifty import isint, isfloat, _exec, LinkFile, warn_once, which, onefile, listfiles, warn_press_key, wopen
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

PS_minimum = np.array([-1.556,
                       -0.111,
                        0.0,
                       -1.947,
                        0.764,
                        0.0,
                       -0.611,
                        0.049,
                        0.0])

class LAMMPS_INTERFACE(object):

    def __init__(self, fn_lmp):
        # Load all LAMMPS commands
        lammps_in = open(fn_lmp).read().splitlines()
        self._lammps_in = lammps_in

        # Neighbor lists checked at every step
        lammps_in.append('neigh_modify delay 0 every 1 check yes')

        # Instantiate
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
        # Our LAMMPS instance
        lmp = self._lmp

        # Update LAMMPS positions 
        self._x_c[:] = x
        lmp.scatter_atoms('x', 2, 3, self._x_c)

        # Update energy and forces in LAMMPS
        lmp.command('run 1 pre yes post no')

        # Get energy and forces from LAMMPS
        V = float(lmp.extract_compute('thermo_pe', 0, 0))

        if force:
            F_c = lmp.gather_atoms('f', 1, 3)
            return V, F_c[:]
        return V

class LAMMPS(Engine):

    """ Engine for carrying out general purpose LAMMPS calculations. """

    def __init__(self, name="lammps", **kwargs):
        ## Keyword args that aren't in this list are filtered out.
        self.valkwd = []
        super(LAMMPS,self).__init__(name=name, **kwargs)

    def setopts(self, **kwargs):
        
        """ Called by __init__ ; Set LAMMPS-specific options and load 
            input file. """
        pass

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory. """

        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        else:
            raise RuntimeError('Coordinate file missing from target \
                                directory.')

    def prepare(self, pbc=False, **kwargs):

        """ Called by __init__ ; prepare the temp directory. """

        self.AtomLists = defaultdict(list)
        self.AtomMask  = []

        ## Extract positions from molecule object
        self.xyz_snapshots = []
        for I in range(len(self.mol)):
            self.xyz_snapshots.append(self.mol.xyzs[I])

        n_atoms = len(self.xyz_snapshots[0])
        if n_atoms == 9:
            self.type = 'trimer'
        elif n_atoms == 6:
            self.type = 'dimer'
        elif n_atoms == 3:
            self.type = 'monomer'
        else:
            raise RuntimeError('Something went wrong in loading the \
                configuration coordinates.')

        ## Copy files for auxiliary lmp engines to be created
        ## within tempdir. Note: lammps_files directory may already
        ## exist in case of an optimization continuation.
        try:
            shutil.copytree(os.path.join(self.srcdir,'lammps_files'),
                os.path.join(self.tempdir,'lammps_files'))
        except OSError:
            logger.info('lammps_files directory already exists in \
                    the working directory. Continuing...\n')
        ## Create all necessary lammps interfaces according
        ## to the data contained in the target
        self.create_interfaces()
   
    def create_interfaces(self):
        
        root = os.getcwd()
        tempdir = self.tempdir

        if self.type == 'monomer':
            os.chdir(os.path.join(tempdir,'lammps_files','monomer'))
            self._lmp_main = LAMMPS_INTERFACE('in.water') 
            os.chdir(root)
        elif self.type == 'dimer':
            os.chdir(os.path.join(tempdir,'lammps_files','dimer'))
            self._lmp_main = LAMMPS_INTERFACE('in.water')

            os.chdir(os.path.join(tempdir,'lammps_files','monomer'))
            self._lmp_monomer = LAMMPS_INTERFACE('in.water')
            os.chdir(root)
        else:
            os.chdir(os.path.join(tempdir,'lammps_files','trimer'))
            self._lmp_main = LAMMPS_INTERFACE('in.water')
            
            os.chdir(os.path.join(tempdir,'lammps_files','dimer'))
            self._lmp_dimer = LAMMPS_INTERFACE('in.water')

            os.chdir(os.path.join(tempdir,'lammps_files','monomer'))
            self._lmp_monomer = LAMMPS_INTERFACE('in.water') 
            os.chdir(root)

    def evaluate_(self, force=False):

        """ 
        Utility function for computing energy and forces using LAMMPS
        """
        Result = {}
        snaps = self.xyz_snapshots
        e_primary = np.array([self._lmp_main.get_V(coords.reshape((1,-1))[0]) \
                              for coords in snaps])

        if self.type == 'monomer':
            # '1-body' energy. we subtract the energy of the structure
            # minimized in the Partridge-Schwenke potential.
            e_series_final = e_primary - self.PS_energy

        elif self.type == 'dimer':
            lmp_m = self._lmp_monomer
            e_m1 = np.array([lmp_m.get_V(coords[:3].reshape((1,-1))[0]) \
                             for coords in snaps])
            e_m2 = np.array([lmp_m.get_V(coords[3:].reshape((1,-1))[0]) \
                             for coords in snaps]) 
            # 2-body energy
            e_series_final = e_primary - (e_m1 + e_m2)

        else:
            lmp_m = self._lmp_monomer
            lmp_d = self._lmp_dimer
            e_m1 = np.array([lmp_m.get_V(coords[:3].reshape((1,-1))[0]) \
                             for coords in snaps])
            e_m2 = np.array([lmp_m.get_V(coords[3:6].reshape((1,-1))[0]) \
                             for coords in snaps])
            e_m3 = np.array([lmp_m.get_V(coords[6:].reshape((1,-1))[0]) \
                             for coords in snaps]) 

            def two_body_series(lmp_d, e_m1, e_m2, dimer_snaps):
                e_pair = np.array([lmp_d.get_V(coords.reshape((1,-1))[0]) \
                                   for coords in dimer_snaps])
                return e_pair - (e_m1 + e_m2)

            d_snaps_1_2 = [coords[:6] for coords in snaps] 
            d_snaps_2_3 = [coords[3:] for coords in snaps]
            d_snaps_1_3 = [np.vstack((coords[:3],coords[6:])) for coords in snaps]
            e_2b_1 = two_body_series(lmp_d, e_m1, e_m2, d_snaps_1_2)
            e_2b_2 = two_body_series(lmp_d, e_m2, e_m3, d_snaps_2_3)
            e_2b_3 = two_body_series(lmp_d, e_m1, e_m3, d_snaps_1_3)
            # 3-body energy
            e_series_final = e_primary - (e_2b_1 + e_2b_2 + e_2b_3 + e_m1 + e_m2 + e_m3)

        KCAL_MOL_TO_KJ_MOL = 4.184 
        Result['Energy'] = e_series_final * KCAL_MOL_TO_KJ_MOL
        if force:
            pass
        return Result

    def energy(self):

        """ Computes the energy using LAMMPS over a trajectory. """

        # TODO: Think about this. 
        # Make sure the forcefield written to the iter_000d directory is
        # copied over to the lammps_files directories and instantiate 
        # new lammps interfaces.
        # cwd at this point is the iter_000d directory
        tempdir = self.tempdir
        abs_run_dir = os.getcwd()
        fn_ff = self.target.FF.fnms[0]
        abs_path_ff = os.path.join(abs_run_dir,fn_ff)
        abs_path_monomer = os.path.join(tempdir,'lammps_files','monomer',fn_ff)
        abs_path_dimer = os.path.join(tempdir,'lammps_files','dimer',fn_ff)
        abs_path_trimer = os.path.join(tempdir,'lammps_files','trimer',fn_ff)

        if self.type == 'trimer':
            shutil.copy(abs_path_ff,abs_path_trimer)
            shutil.copy(abs_path_ff,abs_path_dimer)
            shutil.copy(abs_path_ff,abs_path_monomer)
        elif self.type == 'dimer':
            shutil.copy(abs_path_ff,abs_path_dimer)
            shutil.copy(abs_path_ff,abs_path_monomer)
        else:
            shutil.copy(abs_path_ff,abs_path_monomer)

        self.create_interfaces()

        # Need to calculate a new offset each time 
        # we write a new force field file. 
        if self.type == 'monomer':
            self.PS_energy = self._lmp_main.get_V(PS_minimum)

        if hasattr(self, 'xyz_snapshots'): 
            # Convert calculated energies to kJ/mol
            return self.evaluate_()['Energy'] 
        else:
            raise RuntimeError('Configuration snapshots \
                    not present for target.')

class AbInitio_LAMMPS(AbInitio):

    """Subclass of Target for force and energy matching
    using LAMMPS."""

    def __init__(self,options,tgt_opts,forcefield):
        ## Coordinate file.
        self.set_option(tgt_opts,'coords',default='all.gro')
        ## PDB file for topology (if different from coordinate file.)
        #self.set_option(tgt_opts, 'pdb')
        ## AMBER home directory.
        #self.set_option(options, 'amberhome')
        ## AMBER home directory.
        #self.set_option(tgt_opts, 'amber_leapcmd', 'leapcmd')
        ## Nonbonded cutoff for AMBER (pacth).
        #self.set_option(options, 'amber_nbcut', 'nbcut')
        ## Name of the engine.
        self.engine_ = LAMMPS
        ## Initialize base class.
        super(AbInitio_LAMMPS,self).__init__(options,tgt_opts,forcefield)

