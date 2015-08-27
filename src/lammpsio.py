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
import lammps, shutil
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

    def get_V(self, x):
        # Our LAMMPS instance
        lmp = self._lmp

        # Update LAMMPS positions 
        self._x_c[:] = x
        lmp.scatter_atoms('x', 2, 3, self._x_c)

        # Update energy and forces in LAMMPS
        lmp.command('run 1 pre yes post no')

        # Get energy and forces from LAMMPS
        V = float(lmp.extract_compute('thermo_pe', 0, 0))
        #F_c = lmp.gather_atoms('f', 1, 3)
        #self._dV_dx[:]  = F_c[:]
        #self._dV_dx[:] *= -1.0
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

        num_atoms = len(self.xyz_snapshots[0])
        if num_atoms == 9:
            self.type = 'trimer'
        elif num_atoms == 6:
            self.type = 'dimer'
        elif num_atoms == 3:
            self.type = 'monomer'
        else:
            raise RuntimeError('Something went wrong in loading the \
                configuration coordinates.')

        ## Copy files for auxiliary lmp engines to be created
        ## within tempdir
        shutil.copytree(os.path.join(self.srcdir,'lammps_files'),
                os.path.join(self.tempdir,'lammps_files'))

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

        if self.type == 'monomer':
            e_series = []

            for coords in self.xyz_snapshots:
                e_series.append(self._lmp_main.get_V(coords.reshape((1,-1))[0]) + 248.739362825)

        elif self.type == 'dimer':
            e_d  = []
            e_m1 = []
            e_m2 = []

            for coords in self.xyz_snapshots:
                h2o1_coords = coords[:3]
                h2o2_coords = coords[3:]
                # Energy we want is the 2-body energy, so compute
                # each monomer separately.
                e_m1.append(self._lmp_monomer.get_V(h2o1_coords.reshape((1,-1))[0]))
                e_m2.append(self._lmp_monomer.get_V(h2o2_coords.reshape((1,-1))[0]))
                # Dimer
                e_d.append(self._lmp_main.get_V(coords.reshape((1,-1))[0]))

            e_m1, e_m2, e_d = np.array(e_m1), np.array(e_m2), np.array(e_d)
            e_series = e_d - (e_m1 + e_m2)

        else:
            Es_3B = []
 
            def calc_two_body(dim_eng, dim_coords, E_tot_1_body):
                e_dimer = dim_eng.get_V(dim_coords)
                return e_dimer - E_tot_1_body
            
            for coords in self.xyz_snapshots:
                E_2_body_total = 0.0
                h2o_1_coords = coords[:3].reshape((1,-1))[0]
                h2o_2_coords = coords[3:6].reshape((1,-1))[0]
                h2o_3_coords = coords[6:].reshape((1,-1))[0]
                pair_1 = coords[:6].reshape((1,-1))[0]
                pair_2 = coords[3:].reshape((1,-1))[0]
                pair_3 = np.vstack((coords[:3],coords[6:])).reshape((1,-1))[0]
                Es_1_body = []
                Es_1_body.append(self._lmp_monomer.get_V(h2o_1_coords))
                Es_1_body.append(self._lmp_monomer.get_V(h2o_2_coords))
                Es_1_body.append(self._lmp_monomer.get_V(h2o_3_coords))
                E_2_body_total += calc_two_body(self._lmp_dimer, pair_1, sum(Es_1_body[:2])) + \
                                  calc_two_body(self._lmp_dimer, pair_2, sum(Es_1_body[1:])) + \
                                  calc_two_body(self._lmp_dimer, pair_3, sum([Es_1_body[i] for i in [0,2]]))  
                E_trimer = self._lmp_main.get_V(coords.reshape((1,-1))[0])
                E_3_body = (E_trimer - E_2_body_total) - sum(Es_1_body)
                Es_3B.append(E_3_body)
            e_series = Es_3B
        
        Result['Energy'] = np.array(e_series)
        if force:
            pass
        return Result

    def energy(self):

        """ Computes the energy using AMBER over a trajectory. """

        # TODO: Think about this. 
        # Make sure the forcefield written to the iter_000d directory is
        # copied over to the lammps_files directories and instantiate 
        # new lammps interfaces.
        # cwd at this point is the iter_000d directory
        abs_run_dir = os.getcwd()
        fn_ff = self.target.FF.fnms[0]
        abs_path_ff = os.path.join(abs_run_dir,fn_ff)
        abs_path_monomer = os.path.join(abs_run_dir,'..','lammps_files','monomer',fn_ff)
        abs_path_dimer = os.path.join(abs_run_dir,'..','lammps_files','dimer',fn_ff)
        abs_path_trimer = os.path.join(abs_run_dir,'..','lammps_files','trimer',fn_ff)

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

        if hasattr(self, 'xyz_snapshots'): 
            # Convert calculated energies to kJ/mol
            return self.evaluate_()['Energy'] * 4.184
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

