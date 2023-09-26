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
from forcebalance.vibration import Vibration
from forcebalance.hessian import Hessian
from forcebalance.opt_geo_target import OptGeoTarget
from forcebalance.torsion_profile import TorsionProfileTarget
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
from collections import OrderedDict, defaultdict, Counter
from forcebalance.output import getLogger
from forcebalance.openmmio import OpenMM, UpdateSimulationParameters
import json

logger = getLogger(__name__)
try:
    try:
        # Try importing openmm using >=7.6 namespace
        from openmm.app import *
        from openmm import *
        from openmm.unit import *
        import openmm._openmm as _openmm
    except ImportError:
        # Try importing openmm using <7.6 namespace
        from simtk.openmm.app import *
        from simtk.openmm import *
        from simtk.unit import *
        import simtk.openmm._openmm as _openmm
except:
    pass

try:
    # import the hack for openff.toolkit to improve performance by 10x
    from forcebalance import smirnoff_hack
    # Import the SMIRNOFF forcefield engine and some useful tools
    from openff.toolkit.typing.engines.smirnoff import ForceField as OpenFF_ForceField
    # QYD: name of class are modified to avoid colliding with ForceBalance Molecule
    from openff.toolkit.topology import Molecule as OffMolecule
    from openff.toolkit.topology import Topology as OffTopology
    from openff.units import unit
    from openff.units.openmm import ensure_quantity
    toolkit_import_success = True
except ImportError:
    toolkit_import_success = False
    pass

## pdict is a useless variable if the force field is XML.
pdict = "XML_Override"

def smirnoff_analyze_parameter_coverage(forcefield, tgt_opts):
    printcool("SMIRNOFF Parameter Coverage Analysis")
    assert hasattr(forcefield, 'offxml'), "Only SMIRNOFF Force Field is supported"
    parameter_assignment_data = defaultdict(list)
    parameter_counter = Counter()
    # The openff.toolkit.typing.engines.smirnoff.ForceField object should now be contained in forcebalance.forcefield.FF
    ff = forcefield.openff_forcefield
    # analyze each target
    for tgt_option in tgt_opts:
        target_path = os.path.join('targets', tgt_option['name'])
        # aggregate mol2 file paths from all targets
        mol2_paths = []
        if tgt_option['type'] == 'OPTGEOTARGET_SMIRNOFF':
            # parse optgeo_options_txt and get the names of the mol2 files
            optgeo_options_txt = os.path.join(target_path, tgt_option['optgeo_options_txt'])
            sys_opts = forcebalance.opt_geo_target.OptGeoTarget.parse_optgeo_options(optgeo_options_txt)
            mol2_paths = [os.path.join(target_path,fnm) for sysopt in sys_opts.values() for fnm in sysopt['mol2']]
        elif tgt_option['type'].endswith('_SMIRNOFF'):
            mol2_paths = [os.path.join(target_path,fnm) for fnm in tgt_option['mol2']]
        # analyze SMIRKs terms
        for mol_fnm in mol2_paths:
            # we work with one file at a time to avoid the topology sliently combine "same" molecules
            openff_mol = OffMolecule.from_file(mol_fnm)
            off_topology = OffTopology.from_molecules([openff_mol])
            molecule_force_list = ff.label_molecules(off_topology)
            for mol_idx, mol_forces in enumerate(molecule_force_list):
                for force_tag, force_dict in mol_forces.items():
                    # e.g. force_tag = 'Bonds'
                    for atom_indices, parameters in force_dict.items():

                        if not isinstance(parameters, list):
                            parameters = [parameters]

                        for parameter in parameters:

                            param_dict = {'id': parameter.id, 'smirks': parameter.smirks, 'type':force_tag, 'atoms': list(atom_indices),}
                            parameter_assignment_data[mol_fnm].append(param_dict)
                            parameter_counter[parameter.smirks] += 1
    # write out parameter assignment data
    out_json_path = os.path.join(forcefield.root, 'smirnoff_parameter_assignments.json')
    with open(out_json_path, 'w') as jsonfile:
        json.dump(parameter_assignment_data, jsonfile, indent=2)
        logger.info("Force field assignment data written to %s\n" % out_json_path)
    # print parameter coverages
    logger.info("%4s %-100s   %10s\n" % ("idx", "Parameter", "Count"))
    logger.info("-"*118 + '\n')
    n_covered = 0
    for i,p in enumerate(forcefield.plist):
        smirks = p.split('/')[-1]
        logger.info('%4i %-100s : %10d\n' % (i, p, parameter_counter[smirks]))
        if parameter_counter[smirks] > 0:
            n_covered += 1
    logger.info("SMIRNOFF Parameter Coverage Analysis result: %d/%d parameters are covered.\n" % (n_covered, len(forcefield.plist)))
    logger.info("-"*118 + '\n')

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


def assign_openff_parameter(ff, new_value, pid):
    """
    Assign a SMIRNOFF parameter given the OpenFF ForceField object, the desired parameter value,
    and the parameter's unique ID.
    """
    # Split the parameter's unique ID into four fields using a slash:
    # Input: ProperTorsions/Proper/k1/[*:1]~[#6X3:2]:[#6X3:3]~[*:4]
    # Output: ProperTorsions, Proper, k1, [*:1]~[#6X3:2]:[#6X3:3]~[*:4]
    # The first, third and fourth fields will be used for parameter assignment.
    # We use "value_name" to describe names of individual numerical values within a single parameter type
    # e.g. k1 in the above example.

    if pid.startswith("/"):
        # Handle the case were we are optimizing a handler attribute directly such
        # as the 1-4 scaling factor.
        handler_name, value_name = pid[1:].split('/')

        # Get the OpenFF parameter handler.
        parameter_container = ff.get_parameter_handler(handler_name)

    else:
        (handler_name, tag_name, value_name, smirks) = pid.split('/')

        from openff.toolkit.typing.engines.smirnoff import ParameterList

        # Get the OpenFF parameter object

        # Temporary workaround for OpenFF issue #884
        if not isinstance(ff.get_parameter_handler(handler_name).parameters, ParameterList):

            ff.get_parameter_handler(handler_name)._parameters = ParameterList(
                ff.get_parameter_handler(handler_name).parameters
            )

        parameter_container = ff.get_parameter_handler(handler_name).parameters[smirks]

    # Get param_quantity so we can inspect the type and apply units later if appropriate.
    # Also check for a few special cases and handle them individually.
    if hasattr(parameter_container, value_name):
        # Get the quantity of the parameter in the OpenFF forcefield object
        param_quantity = getattr(parameter_container, value_name)

    elif (hasattr(parameter_container, "_cosmetic_attribs") and
          value_name in parameter_container._cosmetic_attribs):

        return
    else:
        raise KeyError(
            "The {} attribute is not supported by the {} handler".format(
                value_name, handler_name
            )
        )

    if hasattr(param_quantity, "units"):
        setattr(parameter_container, value_name, new_value * param_quantity.units)
    else:
        setattr(parameter_container, value_name, new_value)


def smirnoff_update_pgrads(target):
    """
    Updates a targets pgrads based on smirks present in mol2 files

    This can greatly improve gradients evaluation in big optimizations

    Note
    ----
    1. This function assumes the names of the forcefield parameters has the smirks as the last item
    2. This function assumes params only affect the smirks of its own. This might not be true if parameter_eval is used.
    """
    orig_pgrad_set = set(target.pgrad)
    pgrads_set = set()

    # smirks to param_idxs map
    smirks_params_map = defaultdict(list)
    # New code for mapping smirks to mathematical parameter IDs
    for pname in target.FF.pTree:

        # Make sure we compute the gradients of global parameters such as 1-4 scale
        # factors.
        if pname.startswith('/'):
            pgrads_set.update(target.FF.get_mathid(pname))
        else:
            smirks = pname.rsplit('/',maxsplit=1)[-1]

            for pidx in target.FF.get_mathid(pname):
                smirks_params_map[smirks].append(pidx)

    # get the smirks for this target, keep only the pidx corresponding to these smirks
    smirks_counter = target.engine.get_smirks_counter()
    for smirks in smirks_counter:
        if smirks_counter[smirks] > 0:
            pidx_list = smirks_params_map[smirks]
            # update the set of parameters present in this target
            pgrads_set.update(pidx_list)
    # this ensure we do not add any new items into self.pgrad
    pgrads_set.intersection_update(orig_pgrad_set)
    target.pgrad = sorted(list(pgrads_set))


class SMIRNOFF(OpenMM):

    """ Derived from Engine object for carrying out OpenMM calculations that use the SMIRNOFF force field. """

    def __init__(self, name="openmm", **kwargs):
        self.valkwd = ['ffxml', 'pdb', 'mol2', 'platname', 'precision', 'mmopts', 'vsite_bonds', 'implicit_solvent', 'restrain_k', 'freeze_atoms']
        if not toolkit_import_success:
            warn_once("Note: Failed to import the OpenFF Toolkit - SMIRNOFF Engine will not work. ")
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

        pdbfnm = None
        # Determine the PDB file name.
        if 'pdb' in kwargs and os.path.exists(kwargs['pdb']):
            # Case 1. The PDB file name is provided explicitly
            pdbfnm = kwargs['pdb']
            if not os.path.exists(pdbfnm):
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

        self.mol2 = kwargs.get('mol2')
        if self.mol2:
            for fnm in self.mol2:
                if not os.path.exists(fnm):
                    if hasattr(self, 'FF') and fnm in self.FF.fnms: continue
                    logger.error("%s doesn't exist" % fnm)
                    raise RuntimeError
        else:
            logger.error("Must provide a list of .mol2 files.\n")

        if pdbfnm is not None:
            self.abspdb = os.path.abspath(pdbfnm)
            mpdb = Molecule(pdbfnm)
            for i in ["chain", "atomname", "resid", "resname", "elem"]:
                self.mol.Data[i] = mpdb.Data[i]

        # Store a separate copy of the molecule for reference restraint positions.
        self.ref_mol = deepcopy(self.mol)

    @staticmethod
    def _openff_to_openmm_topology(openff_topology):
        """Convert an OpenFF topology to an OpenMM topology. Currently this requires
        manually adding the v-sites as OpenFF currently does not."""

        from openff.toolkit.topology import TopologyAtom

        openmm_topology = openff_topology.to_openmm()

        # Return the topology if the number of OpenMM particles matches the number
        # expected by the OpenFF topology. This may happen if there are no virtual sites
        # in the system OR if a new version of the the OpenFF toolkit includes virtual
        # sites in the OpenMM topology it returns.
        if openmm_topology.getNumAtoms() == openff_topology.n_topology_particles:
            return openmm_topology

        openmm_chain = openmm_topology.addChain()
        openmm_residue = openmm_topology.addResidue("", chain=openmm_chain)

        for particle in openff_topology.topology_particles:

            if isinstance(particle, TopologyAtom):
                continue

            openmm_topology.addAtom(
                particle.virtual_site.name, app.Element.getByMass(0), openmm_residue
            )
            
        return openmm_topology

    def prepare(self, pbc=False, mmopts={}, **kwargs):

        """
        Prepare the calculation.  Note that we don't create the
        Simulation object yet, because that may depend on MD
        integrator parameters, thermostat, barostat etc.

        This is mostly copied and modified from openmmio.py's OpenMM.prepare(),
        but we are calling ForceField() from the OpenFF toolkit and ignoring
        AMOEBA stuff.
        """

        if hasattr(self, 'abspdb'):
            self.pdb = PDBFile(self.abspdb)
        else:
            pdb1 = "%s-1.pdb" % os.path.splitext(os.path.basename(self.mol.fnm))[0]
            self.mol[0].write(pdb1)
            self.pdb = PDBFile(pdb1)
            os.unlink(pdb1)

        # Create the OpenFF ForceField object.
        if hasattr(self, 'FF'):
            self.offxml = [self.FF.offxml]
            self.forcefield = self.FF.openff_forcefield
        else:
            self.offxml = listfiles(kwargs.get('offxml'), 'offxml', err=True)
            self.forcefield = OpenFF_ForceField(*self.offxml, load_plugins=True)

        ## Load mol2 files for smirnoff topology
        openff_mols = []
        for fnm in self.mol2:
            try:
                mol = OffMolecule.from_file(fnm)
            except Exception as e:
                logger.error("Error when loading %s" % fnm)
                raise e
            openff_mols.append(mol)
        self.off_topology = OffTopology.from_openmm(self.pdb.topology, unique_molecules=openff_mols)

        ## OpenMM options for setting up the System.
        self.mmopts = dict(mmopts)

        ## Specify frozen atoms and restraint force constant
        if 'restrain_k' in kwargs:
            self.restrain_k = kwargs['restrain_k']
        if 'freeze_atoms' in kwargs:
            self.freeze_atoms = kwargs['freeze_atoms'][:]

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
        ## print warning for 'nonbonded_cutoff' keywords
        if 'nonbonded_cutoff' in kwargs:
            logger.warning("nonbonded_cutoff keyword ignored because it's set in the offxml file\n")

        # Apply the FF parameters to the system. Currently this is the only way to
        # determine if the FF will apply virtual sites to the system.
        interchange = self.forcefield.create_interchange(self.off_topology)

        n_virtual_sites = 0
        self._has_virtual_sites = False
        if 'VirtualSites' in interchange.handlers:
            n_virtual_sites = len(interchange['VirtualSites'].slot_map)
            if n_virtual_sites > 0:
                self._has_virtual_sites = True

        ## Generate OpenMM-compatible positions
        self.xyz_omms = []

        for I in range(len(self.mol)):
            xyz = self.mol.xyzs[I]
            xyz_omm = (
                              [Vec3(i[0], i[1], i[2]) for i in xyz]
                              # Add placeholder positions for an v-sites.
                              + [Vec3(0.0, 0.0, 0.0)] * n_virtual_sites
                      ) * angstrom

            if self.pbc:
                # Obtain the periodic box
                if self.mol.boxes[I].alpha != 90.0 or self.mol.boxes[I].beta != 90.0 or self.mol.boxes[
                    I].gamma != 90.0:
                    logger.error('OpenMM cannot handle nonorthogonal boxes.\n')
                    raise RuntimeError
                box_omm = np.diag([self.mol.boxes[I].a, self.mol.boxes[I].b, self.mol.boxes[I].c]) * angstrom
            else:
                box_omm = None
            # Finally append it to list.
            self.xyz_omms.append((xyz_omm, box_omm))

        openmm_topology = interchange.to_openmm_topology()
        openmm_positions = (
                                   self.pdb.positions.value_in_unit(angstrom) +
                                   # Add placeholder positions for an v-sites.
                                   [Vec3(0.0, 0.0, 0.0)] * n_virtual_sites
                           ) * angstrom
        self.mod = app.modeller.Modeller(openmm_topology, openmm_positions)

        ## Build a topology and atom lists.
        Top = self.mod.getTopology()
        Atoms = list(Top.atoms())

        # vss = [(i, [system.getVirtualSite(i).getParticle(j) for j in range(system.getVirtualSite(i).getNumParticles())]) \
        #            for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
        self.AtomLists = defaultdict(list)
        self.AtomLists['Mass'] = [a.element.mass.value_in_unit(dalton) if a.element is not None else 0 for a in Atoms]
        self.AtomLists['ParticleType'] = ['A' if m >= 1.0 else 'D' for m in self.AtomLists['Mass']]
        self.AtomLists['ResidueNumber'] = [a.residue.index for a in Atoms]
        self.AtomMask = [a == 'A' for a in self.AtomLists['ParticleType']]
        self.realAtomIdxs = [i for i, a in enumerate(self.AtomMask) if a is True]
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

        # Because self.forcefield is being updated in forcebalance.forcefield.FF.make()
        # there is no longer a need to create a new force field object here.
        try:
            interchange = self.forcefield.create_interchange(self.off_topology)
            self.system = interchange.to_openmm()
            self.off_topology = interchange.topology
        except Exception as error:
            logger.error("Error when creating system for %s" % self.mol2)
            raise error
        # Commenting out all virtual site stuff for now.
        # self.vsinfo = PrepareVirtualSites(self.system)
        self.nbcharges = np.zeros(self.system.getNumParticles())

        #----
        # If the virtual site parameters have changed,
        # the simulation object must be remade.
        #----
        # vsprm = GetVirtualSiteParameters(self.system)
        # if hasattr(self,'vsprm') and len(self.vsprm) > 0 and np.max(np.abs(vsprm - self.vsprm)) != 0.0:
        #     if hasattr(self, 'simulation'):
        #         delattr(self, 'simulation')
        # self.vsprm = vsprm.copy()

        has_vsites = False
        for particle_idx in range(self.system.getNumParticles()):
            if self.system.isVirtualSite(particle_idx):
                has_vsites = True

        if has_vsites:
            raise Exception("ForceBalance can't currently handle SMIRNOFF vsites. "
                            "Downgrade to ForceBalance 1.9.3 or earlier to handle those.")

        if hasattr(self, 'simulation'):
            UpdateSimulationParameters(self.system, self.simulation)
        else:
            self.create_simulation(**self.simkwargs)

    def _update_positions(self, X1, disable_vsite):
        # X1 is a numpy ndarray not vec3

        if disable_vsite:
            super(SMIRNOFF, self)._update_positions(X1, disable_vsite)
            return

        n_v_sites = (
            self.mod.getTopology().getNumAtoms() - self.pdb.topology.getNumAtoms()
        )

        # Add placeholder positions for an v-sites.
        if isinstance(X1, np.ndarray):
            X1 = numpy.vstack([X1, np.zeros((n_v_sites, 3))]) * angstrom
        else:
            X1 = (X1 + [Vec3(0.0, 0.0, 0.0)] * n_v_sites) * angstrom

        self.simulation.context.setPositions(X1)
        self.simulation.context.computeVirtualSites()

    def interaction_energy(self, fraga, fragb):

        """
        Calculate the interaction energy for two fragments.
        Because this creates two new objects and requires passing in the mol2 argument,
        the codes are copied and modified from the OpenMM class.
        """

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

    def get_smirks_counter(self):
        """Get a counter for the time of appreance of each SMIRKS"""
        smirks_counter = Counter()
        molecule_force_list = self.forcefield.label_molecules(self.off_topology)
        for mol_idx, mol_forces in enumerate(molecule_force_list):
            for force_tag, force_dict in mol_forces.items():
                # e.g. force_tag = 'Bonds'
                for parameters in force_dict.values():

                    if not isinstance(parameters, list):
                        parameters = [parameters]

                    for parameter in parameters:
                        smirks_counter[parameter.smirks] += 1

        return smirks_counter

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
        self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
        self.engine_ = SMIRNOFF
        ## Initialize base class.
        super(AbInitio_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)

    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        # we update the self.pgrads here so it's not overwritten in rtarget.py
        smirnoff_update_pgrads(self)

class Vibration_SMIRNOFF(Vibration):
    """ Vibrational frequency matching using using SMIRNOFF format powered by OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="input.pdb")
        self.set_option(tgt_opts,'pdb',default="conf.pdb")
        self.set_option(tgt_opts,'mol2',forceprint=True)
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
        self.engine_ = SMIRNOFF
        ## Initialize base class.
        super(Vibration_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)

    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        # we update the self.pgrads here so it's not overwritten in rtarget.py
        smirnoff_update_pgrads(self)

class Hessian_SMIRNOFF(Hessian):
    """ Internal coordinate Hessian matching using SMIRNOFF format powered by OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="input.pdb")
        self.set_option(tgt_opts,'pdb',default="conf.pdb")
        self.set_option(tgt_opts,'mol2',forceprint=True)
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
        self.engine_ = SMIRNOFF
        ## Initialize base class.
        super(Hessian_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)

    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        # we update the self.pgrads here so it's not overwritten in rtarget.py
        smirnoff_update_pgrads(self)
        
class OptGeoTarget_SMIRNOFF(OptGeoTarget):
    """ Optimized geometry fitting using SMIRNOFF format powered by OpenMM """
    def __init__(self,options,tgt_opts,forcefield):
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
        self.engine_ = SMIRNOFF
        ## Initialize base class.
        super(OptGeoTarget_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)

    def create_engines(self, engine_args):
        """ create a dictionary of self.engines = {sysname: Engine} """
        self.engines = OrderedDict()
        for sysname, sysopt in self.sys_opts.items():
            # SMIRNOFF is a subclass of OpenMM engine but it requires the mol2 input
            # note: OpenMM.mol is a Molecule class instance;  mol2 is a file format.
            # path to .pdb file
            pdbpath = os.path.join(self.root, self.tgtdir, sysopt['topology'])
            # a list of paths to .mol2 files
            mol2path = [os.path.join(self.root, self.tgtdir, f) for f in sysopt['mol2']]
            # use the PDB file with topology
            M = Molecule(os.path.join(self.root, self.tgtdir, sysopt['topology']))
            # replace geometry with values from xyz file for higher presision
            M0 = Molecule(os.path.join(self.root, self.tgtdir, sysopt['geometry']))
            M.xyzs = M0.xyzs
            # here mol=M is given for the purpose of using the topology from the input pdb file
            # if we don't do this, pdb=top.pdb option will only copy some basic information but not the topology into OpenMM.mol (openmmio.py line 615)
            self.engines[sysname] = self.engine_(target=self, mol=M, name=sysname, pdb=pdbpath, mol2=mol2path, **engine_args)
        self.build_system_mval_masks()

    def build_system_mval_masks(self):
        """
        Build a mask of mvals for each system, to speed up finite difference gradients

        Note
        ----
        1. This function assumes the names of the forcefield parameters has the smirks as the last item
        2. This function assumes params only affect the smirks of its own. This might not be true if parameter_eval is used.
        """
        # only need to build once
        if hasattr(self, 'system_mval_masks'): return
        n_params = len(self.FF.map)
        # default mask with all False
        system_mval_masks = {sysname: np.zeros(n_params, dtype=bool) for sysname in self.sys_opts}
        orig_pgrad_set = set(self.pgrad)
        # smirks to param_idxs map
        smirks_params_map = defaultdict(list)
        # New code for mapping smirks to mathematical parameter IDs
        for pname in self.FF.pTree:

            # Make sure we compute the gradients of global parameters such as 1-4 scale
            # factors.
            if pname.startswith('/'):

                for sysname in self.sys_opts:
                    pidx_list = [pidx for pidx in self.FF.get_mathid(pname)]
                    system_mval_masks[sysname][pidx_list] = True

            else:
                smirks = pname.rsplit('/',maxsplit=1)[-1]
                # print("pname %s mathid %s -> smirks %s" % (pname, str(self.FF.get_mathid(pname)), smirks))
                for pidx in self.FF.get_mathid(pname):
                    smirks_params_map[smirks].append(pidx)
        # Old code for mapping smirks to mathematical parameter IDs
        # for pname, pidx in self.FF.map.items():
        #     smirks = pname.rsplit('/',maxsplit=1)[-1]
        #     smirks_params_map[smirks].append(pidx)
        # go over all smirks for each system
        for sysname in self.sys_opts:
            engine = self.engines[sysname]
            smirks_counter = engine.get_smirks_counter()
            for smirks in smirks_counter:
                if smirks_counter[smirks] > 0:
                    pidx_list = smirks_params_map[smirks]
                    # set mask value to True for present smirks
                    system_mval_masks[sysname][pidx_list] = True
        # finish
        logger.info("system_mval_masks is built for faster gradient evaluations")
        self.system_mval_masks = system_mval_masks

class TorsionProfileTarget_SMIRNOFF(TorsionProfileTarget):
    """ Force and energy matching using SMIRKS native Open Force Field (SMIRNOFF). """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'pdb',default="conf.pdb")
        # List of .mol2 files for SMIRNOFF to set up the system
        self.set_option(tgt_opts,'mol2',forceprint=True)
        self.set_option(tgt_opts,'coords',default="scan.xyz")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="Reference", forceprint=True)
        self.engine_ = SMIRNOFF
        ## Initialize base class.
        super(TorsionProfileTarget_SMIRNOFF,self).__init__(options,tgt_opts,forcefield)

    def submit_jobs(self, mvals, AGrad=False, AHess=False):
        # we update the self.pgrads here so it's not overwritten in rtarget.py
        smirnoff_update_pgrads(self)

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
