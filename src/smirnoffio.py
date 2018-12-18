""" @package forcebalance.smirnoff SMIRNOFF force field support.

@author Lee-Ping Wang
@date 12/2018
"""
from __future__ import division

from builtins import zip
from builtins import range
import os
from forcebalance import BaseReader
from forcebalance.abinitio import AbInitio
from forcebalance.binding import BindingEnergy
from forcebalance.liquid import Liquid
from forcebalance.interaction import Interaction
from forcebalance.moments import Moments
from forcebalance.hydration import Hydration
import networkx as nx
import numpy as np
import sys
from forcebalance.finite_difference import *
import pickle
import shutil
from copy import deepcopy
from forcebalance.engine import Engine
from forcebalance.molecule import *
from forcebalance.chemistry import *
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from collections import OrderedDict
from forcebalance.output import getLogger
from forcebalance.openmmio import OpenMM, UpdateSimulationParameters

logger = getLogger(__name__)
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import simtk.openmm._openmm as _openmm
except:
    pass

try:
    # Import the SMIRNOFF forcefield engine and some useful tools
    from openforcefield.typing.engines.smirnoff import ForceField
    # LPW: openforcefield's PME is different from openmm's PME
    from openforcefield.typing.engines.smirnoff.forcefield import NoCutoff, PME
    from openforcefield.utils import get_data_filename, extractPositionsFromOEMol, generateTopologyFromOEMol
    # Import the OpenEye toolkit
    from openeye import oechem
except:
    warn_once("Failed to import openforcefield and/or OpenEye toolkit.")
    
"""Dictionary for building parameter identifiers.  As usual they go like this:
Bond/length/OW.HW
The dictionary is two-layered because the same interaction type (Bond)
could be under two different parent types (HarmonicBondForce, AmoebaHarmonicBondForce)"""
suffix_dict = { "HarmonicBondForce" : {"Bond" : ["class1","class2"]},
                "HarmonicAngleForce" : {"Angle" : ["class1","class2","class3"],},
                "PeriodicTorsionForce" : {"Proper" : ["class1","class2","class3","class4"],},
                "NonbondedForce" : {"Atom": ["type"]},
                "CustomNonbondedForce" : {"Atom": ["class"]},
                "GBSAOBCForce" : {"Atom": ["type"]},
                "AmoebaBondForce" : {"Bond" : ["class1","class2"]},
                "AmoebaAngleForce" : {"Angle" : ["class1","class2","class3"]},
                "AmoebaStretchBendForce" : {"StretchBend" : ["class1","class2","class3"]},
                "AmoebaVdwForce" : {"Vdw" : ["class"]},
                "AmoebaMultipoleForce" : {"Multipole" : ["type","kz","kx"], "Polarize" : ["type"]},
                "AmoebaUreyBradleyForce" : {"UreyBradley" : ["class1","class2","class3"]},
                "Residues.Residue" : {"VirtualSite" : ["index"]},
                ## LPW's custom parameter definitions
                "ForceBalance" : {"GB": ["type"]},
                }

## pdict is a useless variable if the force field is XML.
pdict = "XML_Override"

class SMIRNOFF_Reader(BaseReader):
    """ Class for parsing OpenMM force field files. """
    def __init__(self,fnm):
        ## Initialize the superclass. :)
        super(SMIRNOFF_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict

    def build_pid(self, element, parameter):
        """ Build the parameter identifier (see _link_ for an example)
        @todo Add a link here """
        ParentType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:])
        InteractionType = element.tag
        try:
            Involved = element.attrib["smirks"]
            return "/".join([ParentType, InteractionType, parameter, Involved])
        except:
            logger.info("Minor warning: Parameter ID %s doesn't contain any SMIRKS patterns, redundancies are possible\n" % ("/".join([InteractionType, parameter])))
            return "/".join([ParentType, InteractionType, parameter])

class SMIRNOFF(OpenMM):

    """ Derived from Engine object for carrying out OpenMM calculations that use the SMIRNOFF force field. """

    def __init__(self, name="openmm", **kwargs):
        self.valkwd = ['ffxml', 'pdb', 'mol2', 'platname', 'precision', 'mmopts', 'vsite_bonds', 'implicit_solvent']
        super(SMIRNOFF,self).__init__(name=name, **kwargs)

    def readsrc(self, **kwargs):
        """ 
        SMIRNOFF simulations always require the following passed in via kwargs:

        Parameters
        ----------
        pdb : string
            Name of a .pdb file containing the topology of the system
        mol2 : list
            A list of .mol2 file names containing the molecule/residue templates of the system

        Also provide 1 of the following, containing the coordinates to be used:
        mol : Molecule
            forcebalance.Molecule object
        coords : string
            Name of a file (readable by forcebalance.Molecule)
            This could be the same as the pdb argument from above.
        """

        pdbfnm = kwargs.get('pdb')
        # Determine the PDB file name.
        if not pdbfnm:
            raise RuntimeError('Name of PDB file not provided.')
        elif not os.path.exists(pdbfnm):
            logger.error("%s specified but doesn't exist\n" % pdbfnm)
            raise RuntimeError

        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs:
            if not os.path.exists(kwargs['coords']):
                logger.error("%s specified but doesn't exist\n" % kwargs['coords'])
                raise RuntimeError
            self.mol = Molecule(kwargs['coords'])
        else:
            logger.error('Must provide either a molecule object or coordinate file.\n')
            raise RuntimeError

        # Here we cannot distinguish the .mol2 files linked by the target 
        # vs. the .mol2 files to be provided by the force field.
        # But we can assume that these files should exist when this function is called.
        self.mol2_files = kwargs.get('mol2')
        if self.mol2_files:
            for fnm in self.mol2_files:
                if not os.path.exists(fnm):
                    logger.error("%s doesn't exist" % fnm)
                    raise RuntimeError
        else:
            logger.error("Must provide a list of .mol2 files.\n")

        self.abspdb = os.path.abspath(pdbfnm)
        mpdb = Molecule(pdbfnm)
        for i in ["chain", "atomname", "resid", "resname", "elem"]:
            self.mol.Data[i] = mpdb.Data[i]

    def prepare(self, pbc=False, mmopts={}, **kwargs):

        """
        Prepare the calculation.  Note that we don't create the
        Simulation object yet, because that may depend on MD
        integrator parameters, thermostat, barostat etc.

        This is mostly copied and modified from openmmio.py's OpenMM.prepare(),
        but we are calling ForceField() from the OpenFF toolkit and ignoring 
        AMOEBA stuff.
        """
        self.pdb = PDBFile(self.abspdb)

        ## Create the OpenFF ForceField object.
        if hasattr(self, 'FF'):
            self.offxml = [self.FF.offxml]
            self.forcefield = ForceField(os.path.join(self.root, self.FF.ffdir, self.FF.offxml))
        else:
            self.offxml = listfiles(kwargs.get('offxml'), 'offxml', err=True)
            self.forcefield = ForceField(*self.offxml)

        ## OpenMM options for setting up the System.
        self.mmopts = dict(mmopts)

        ## Set system options from ForceBalance force field options.
        fftmp = False
        if hasattr(self,'FF'):
            self.mmopts['rigidWater'] = self.FF.rigid_water
            if not all([os.path.exists(f) for f in self.FF.fnms]):
                # If the parameter files don't already exist, create them for the purpose of
                # preparing the engine, but then delete them afterward.
                fftmp = True
                self.FF.make(np.zeros(self.FF.np))

        ## Set system options from periodic boundary conditions.
        self.pbc = pbc
        if pbc:
            minbox = min([self.mol.boxes[0].a, self.mol.boxes[0].b, self.mol.boxes[0].c])
            self.SetPME = True
            self.mmopts.setdefault('nonbondedMethod', PME)
            nonbonded_cutoff = kwargs.get('nonbonded_cutoff', 8.5)
            # Conversion to nanometers
            nonbonded_cutoff /= 10
            if nonbonded_cutoff > 0.05*(float(int(minbox - 1))):
                warn_press_key("nonbonded_cutoff = %.1f should be smaller than half the box size = %.1f Angstrom" % (nonbonded_cutoff*10, minbox))

            self.mmopts.setdefault('nonbondedCutoff', nonbonded_cutoff*nanometer)
            self.mmopts.setdefault('useSwitchingFunction', True)
            self.mmopts.setdefault('switchingDistance', (nonbonded_cutoff-0.1)*nanometer)
            self.mmopts.setdefault('useDispersionCorrection', True)
        else:
            if 'nonbonded_cutoff' in kwargs or 'vdw_cutoff' in kwargs:
                warn_press_key('No periodic boundary conditions, your provided nonbonded_cutoff and vdw_cutoff will not be used')
            self.SetPME = False
            self.mmopts.setdefault('nonbondedMethod', NoCutoff)
            self.mmopts['removeCMMotion'] = False

        ## Generate OpenMM-compatible positions
        self.xyz_omms = []
        for I in range(len(self.mol)):
            xyz = self.mol.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # An extra step with adding virtual particles
            mod = Modeller(self.pdb.topology, xyz_omm)
            # LPW commenting out because we don't have virtual sites yet.
            # mod.addExtraParticles(self.forcefield)
            if self.pbc:
                # Obtain the periodic box
                if self.mol.boxes[I].alpha != 90.0 or self.mol.boxes[I].beta != 90.0 or self.mol.boxes[I].gamma != 90.0:
                    logger.error('OpenMM cannot handle nonorthogonal boxes.\n')
                    raise RuntimeError
                box_omm = [Vec3(self.mol.boxes[I].a, 0, 0)*angstrom,
                           Vec3(0, self.mol.boxes[I].b, 0)*angstrom,
                           Vec3(0, 0, self.mol.boxes[I].c)*angstrom]
            else:
                box_omm = None
            # Finally append it to list.
            self.xyz_omms.append((mod.getPositions(), box_omm))

        ## Build a topology and atom lists.
        Top = mod.getTopology()
        Atoms = list(Top.atoms())
        Bonds = [(a.index, b.index) for a, b in list(Top.bonds())]

        # vss = [(i, [system.getVirtualSite(i).getParticle(j) for j in range(system.getVirtualSite(i).getNumParticles())]) \
        #            for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
        self.AtomMask = []
        self.AtomLists = defaultdict(list)
        self.AtomLists['Mass'] = [a.element.mass.value_in_unit(dalton) if a.element is not None else 0 for a in Atoms]
        self.AtomLists['ParticleType'] = ['A' if m >= 1.0 else 'D' for m in self.AtomLists['Mass']]
        self.AtomLists['ResidueNumber'] = [a.residue.index for a in Atoms]
        self.AtomMask = [a == 'A' for a in self.AtomLists['ParticleType']]
        if hasattr(self,'FF') and fftmp:
            for f in self.FF.fnms: 
                os.unlink(f)

    def update_simulation(self, **kwargs):

        """
        Create the simulation object, or update the force field
        parameters in the existing simulation object.  This should be
        run when we write a new force field XML file.
        """
        if len(kwargs) > 0:
            self.simkwargs = kwargs

        self.mod = Modeller(self.pdb.topology, self.pdb.positions)
        self.forcefield = ForceField(*self.offxml)
        # This part requires the OpenEye tools but may be replaced
        # by RDKit when that support comes online.
        oemols = []
        for fnm in self.mol2_files:
            mol = oechem.OEGraphMol()
            ifs = oechem.oemolistream(fnm)
            oechem.OEReadMolecule(ifs, mol)
            oechem.OETriposAtomNames(mol)
            oemols.append(mol)
        self.system = self.forcefield.createSystem(self.pdb.topology, oemols, **self.mmopts)

        # Commenting out all virtual site stuff for now.
        # self.vsinfo = PrepareVirtualSites(self.system)
        self.nbcharges = np.zeros(self.system.getNumParticles())

        for i in self.system.getForces():
            if isinstance(i, NonbondedForce):
                self.nbcharges = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])
                if self.SetPME:
                    i.setNonbondedMethod(i.PME)
            if isinstance(i, AmoebaMultipoleForce):
                if self.SetPME:
                    i.setNonbondedMethod(i.PME)

        #----
        # If the virtual site parameters have changed,
        # the simulation object must be remade.
        #----
        # vsprm = GetVirtualSiteParameters(self.system)
        # if hasattr(self,'vsprm') and len(self.vsprm) > 0 and np.max(np.abs(vsprm - self.vsprm)) != 0.0:
        #     if hasattr(self, 'simulation'):
        #         delattr(self, 'simulation')
        # self.vsprm = vsprm.copy()

        if hasattr(self, 'simulation'):
            UpdateSimulationParameters(self.system, self.simulation)
        else:
            self.create_simulation(**self.simkwargs)

    def optimize(self, shot=0, crit=1e-4):

        """ Optimize the geometry and align the optimized geometry to the starting geometry, and return the RMSD. 
        We are copying because we need to skip over virtual site stuff.
        """

        steps = int(max(1, -1*np.log10(crit)))
        self.update_simulation()
        self.set_positions(shot)
        # Get the previous geometry.
        X0 = np.array([j for i, j in enumerate(self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(angstrom)) if self.AtomMask[i]])
        # Minimize the energy.  Optimizer works best in "steps".
        for logc in np.linspace(0, np.log10(crit), steps):
            self.simulation.minimizeEnergy(tolerance=10**logc*kilojoule/mole)
        # Get the optimized geometry.
        S = self.simulation.context.getState(getPositions=True, getEnergy=True)
        X1 = np.array([j for i, j in enumerate(S.getPositions().value_in_unit(angstrom)) if self.AtomMask[i]])
        E = S.getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
        # Align to original geometry.
        M = deepcopy(self.mol[0])
        M.xyzs = [X0, X1]
        if not self.pbc:
            M.align(center=False)
        X1 = M.xyzs[1]
        # Set geometry in OpenMM, requires some hoops.
        mod = Modeller(self.pdb.topology, [Vec3(i[0],i[1],i[2]) for i in X1]*angstrom)
        # mod.addExtraParticles(self.forcefield)
        # self.simulation.context.setPositions(ResetVirtualSites(mod.getPositions(), self.system))
        self.simulation.context.setPositions(mod.getPositions())
        return E, M.ref_rmsd(0)[1]

    def interaction_energy(self, fraga, fragb):

        """ Calculate the interaction energy for two fragments. """

        self.update_simulation()

        if self.name == 'A' or self.name == 'B':
            logger.error("Don't name the engine A or B!\n")
            raise RuntimeError

        # Create two subengines.
        if hasattr(self,'target'):
            if not hasattr(self,'A'):
                self.A = SMIRNOFF(name="A", mol=self.mol.atom_select(fraga), mol2=self.mol2, target=self.target)
            if not hasattr(self,'B'):
                self.B = SMIRNOFF(name="B", mol=self.mol.atom_select(fragb), mol2=self.mol2, target=self.target)
        else:
            if not hasattr(self,'A'):
                self.A = SMIRNOFF(name="A", mol=self.mol.atom_select(fraga), mol2=self.mol2, platname=self.platname, \
                                  precision=self.precision, offxml=self.offxml, mmopts=self.mmopts)
            if not hasattr(self,'B'):
                self.B = SMIRNOFF(name="B", mol=self.mol.atom_select(fragb), mol2=self.mol2, platname=self.platname, \
                                  precision=self.precision, offxml=self.offxml, mmopts=self.mmopts)

        # Interaction energy needs to be in kcal/mol.
        D = self.energy()
        A = self.A.energy()
        B = self.B.energy()

        return (D - A - B) / 4.184

class Liquid_SMIRNOFF(Liquid):
    """ Condensed phase property matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'force_cuda',forceprint=True)
        # Enable multiple timestep integrator
        self.set_option(tgt_opts,'mts_integrator',forceprint=True)
        # Enable ring polymer MD
        self.set_option(options,'rpmd_beads',forceprint=True)
        # List of .mol2 files for SMIRNOFF to set up the system
        self.set_option(tgt_opts,'mol2',forceprint=True)
        # OpenMM precision
        self.set_option(tgt_opts,'openmm_precision','precision',default="mixed")
        # OpenMM platform
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA")
        # Name of the liquid coordinate file.
        self.set_option(tgt_opts,'liquid_coords',default='liquid.pdb',forceprint=True)
        # Name of the gas coordinate file.
        self.set_option(tgt_opts,'gas_coords',default='gas.pdb',forceprint=True)
        # Name of the surface tension coordinate file. (e.g. an elongated box with a film of water)
        self.set_option(tgt_opts,'nvt_coords',default='surf.pdb',forceprint=True)
        # Set the number of steps between MC barostat adjustments.
        self.set_option(tgt_opts,'mc_nbarostat')
        # Class for creating engine object.
        self.engine_ = SMIRNOFF
        # Name of the engine to pass to npt.py.
        self.engname = "smirnoff"
        # Command prefix.
        self.nptpfx = "bash runcuda.sh"
        if tgt_opts['remote_backup']:
            self.nptpfx += " -b"
        # Extra files to be linked into the temp-directory.
        self.nptfiles = []
        self.nvtfiles = []
        # Set some options for the polarization correction calculation.
        self.gas_engine_args = {}
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ['runcuda.sh']
        # Initialize the base class.
        super(Liquid_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output = ['liquid-md.pdb', 'liquid-md.dcd']
        # These functions need to be called after self.nptfiles is populated
        self.post_init(options)

class AbInitio_SMIRNOFF(AbInitio):
    """ Force and energy matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'pdb',default="conf.pdb")
        # List of .mol2 files for SMIRNOFF to set up the system
        self.set_option(tgt_opts,'mol2',forceprint=True)
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
        self.engine_ = SMIRNOFF
        ## Initialize base class.
        super(AbInitio_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)

# class BindingEnergy_SMIRNOFF(BindingEnergy):
#     """ Binding energy matching using OpenMM. """

#     def __init__(self,options,tgt_opts,forcefield):
#         self.engine_ = OpenMM
#         self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
#         self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
#         ## Initialize base class.
#         super(BindingEnergy_OpenMM,self).__init__(options,tgt_opts,forcefield)

# class Interaction_SMIRNOFF(Interaction):
#     """ Interaction matching using OpenMM. """
#     def __init__(self,options,tgt_opts,forcefield):
#         ## Default file names for coordinates and key file.
#         self.set_option(tgt_opts,'coords',default="all.pdb")
#         self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
#         self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
#         self.engine_ = OpenMM
#         ## Initialize base class.
#         super(Interaction_OpenMM,self).__init__(options,tgt_opts,forcefield)

# class Moments_SMIRNOFF(Moments):
#     """ Multipole moment matching using OpenMM. """
#     def __init__(self,options,tgt_opts,forcefield):
#         ## Default file names for coordinates and key file.
#         self.set_option(tgt_opts,'coords',default="input.pdb")
#         self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
#         self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
#         self.engine_ = OpenMM
#         ## Initialize base class.
#         super(Moments_OpenMM,self).__init__(options,tgt_opts,forcefield)

# class Hydration_SMIRNOFF(Hydration):
#     """ Single point hydration free energies using OpenMM. """

#     def __init__(self,options,tgt_opts,forcefield):
#         ## Default file names for coordinates and key file.
#         # self.set_option(tgt_opts,'coords',default="input.pdb")
#         self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
#         self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
#         self.engine_ = SMIRNOFF
#         self.engname = "smirnoff"
#         ## Scripts to be copied from the ForceBalance installation directory.
#         self.scripts = ['runcuda.sh']
#         ## Suffix for coordinate files.
#         self.crdsfx = '.pdb'
#         ## Command prefix.
#         self.prefix = "bash runcuda.sh"
#         if tgt_opts['remote_backup']:
#             self.prefix += " -b"
#         ## Initialize base class.
#         super(Hydration_OpenMM,self).__init__(options,tgt_opts,forcefield)
#         ## Send back the trajectory file.
#         if self.save_traj > 0:
#             self.extra_output = ['openmm-md.dcd']
