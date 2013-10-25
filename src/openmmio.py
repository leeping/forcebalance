""" @package forcebalance.openmmio OpenMM input/output.

@author Lee-Ping Wang
@date 04/2012
"""

import os
from forcebalance import BaseReader
from forcebalance.abinitio import AbInitio
from forcebalance.liquid import Liquid
from forcebalance.interaction import Interaction
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
logger = getLogger(__name__)
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import simtk.openmm._openmm as _openmm
except:
    pass

def energy_components(Sim, verbose=False):
    # Before using EnergyComponents, make sure each Force is set to a different group.
    EnergyTerms = OrderedDict()
    Potential = Sim.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
    Kinetic = Sim.context.getState(getEnergy=True).getKineticEnergy() / kilojoules_per_mole
    for i in range(Sim.system.getNumForces()):
        EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
    EnergyTerms['Potential'] = Potential
    EnergyTerms['Kinetic'] = Kinetic
    EnergyTerms['Total'] = Potential+Kinetic
    return EnergyTerms

def get_multipoles(simulation,q=None,positions=None):
    """Return the current multipole moments in Debye and Buckingham units. """
    dx = 0.0
    dy = 0.0
    dz = 0.0
    qxx = 0.0
    qxy = 0.0
    qxz = 0.0
    qyy = 0.0
    qyz = 0.0
    qzz = 0.0
    enm_debye = 48.03204255928332 # Conversion factor from e*nm to Debye
    for i in simulation.system.getForces():
        if i.__class__.__name__ == "AmoebaMultipoleForce":
            mm = i.getSystemMultipoleMoments(simulation.context)
            dx += mm[1]
            dy += mm[2]
            dz += mm[3]
            qxx += mm[4]
            qxy += mm[5]
            qxz += mm[6]
            qyy += mm[8]
            qyz += mm[9]
            qzz += mm[12]
        if i.__class__.__name__ == "NonbondedForce":
            # Get array of charges.
            if q == None:
                q = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])
            # Get array of positions in nanometers.
            if positions == None:
                positions = simulation.context.getState(getPositions=True).getPositions()
            x = np.array(positions.value_in_unit(nanometer))
            xx, xy, xz, yy, yz, zz = (x[:,i]*x[:,j] for i, j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)])
            # Multiply charges by positions to get dipole moment.
            dip = enm_debye * np.sum(x*q.reshape(-1,1),axis=0)
            dx += dip[0]
            dy += dip[1]
            dz += dip[2]
            qxx += enm_debye * 10 * np.sum(q*xx)
            qxy += enm_debye * 10 * np.sum(q*xy)
            qxz += enm_debye * 10 * np.sum(q*xz)
            qyy += enm_debye * 10 * np.sum(q*yy)
            qyz += enm_debye * 10 * np.sum(q*yz)
            qzz += enm_debye * 10 * np.sum(q*zz)
            tr = qxx+qyy+qzz
            qxx -= tr/3
            qyy -= tr/3
            qzz -= tr/3
    # This ordering has to do with the way TINKER prints it out.
    return [dx,dy,dz,qxx,qxy,qyy,qxz,qyz,qzz]

def get_dipole(simulation,q=None,positions=None):
    """Return the current dipole moment in Debye.
    Note that this quantity is meaningless if the system carries a net charge."""
    return get_multipoles(simulation, q=q, positions=positions)[:3]

def ResetVirtualSites(positions, system):
    """Given a set of OpenMM-compatible positions and a System object,
    compute the correct virtual site positions according to the System."""
    if any([system.isVirtualSite(i) for i in range(system.getNumParticles())]):
        pos = positions.value_in_unit(nanometer)
        for i in range(system.getNumParticles()):
            if system.isVirtualSite(i):
                vs = system.getVirtualSite(i)
                vstype = vs.__class__.__name__
                # Faster code to work around Python API slowness.
                if vstype == 'TwoParticleAverageSite':
                    vspos = _openmm.TwoParticleAverageSite_getWeight(vs, 0)*pos[_openmm.VirtualSite_getParticle(vs, 0)] \
                        + _openmm.TwoParticleAverageSite_getWeight(vs, 1)*pos[_openmm.VirtualSite_getParticle(vs, 1)]
                elif vstype == 'ThreeParticleAverageSite':
                    vspos = _openmm.ThreeParticleAverageSite_getWeight(vs, 0)*pos[_openmm.VirtualSite_getParticle(vs, 0)] \
                        + _openmm.ThreeParticleAverageSite_getWeight(vs, 1)*pos[_openmm.VirtualSite_getParticle(vs, 1)] \
                        + _openmm.ThreeParticleAverageSite_getWeight(vs, 2)*pos[_openmm.VirtualSite_getParticle(vs, 2)]
                elif vstype == 'OutOfPlaneSite':
                    v1 = pos[_openmm.VirtualSite_getParticle(vs, 1)] - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    v2 = pos[_openmm.VirtualSite_getParticle(vs, 2)] - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    cross = Vec3(v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0])
                    vspos = pos[_openmm.VirtualSite_getParticle(vs, 0)] + _openmm.OutOfPlaneSite_getWeight12(vs)*v1 \
                        + _openmm.OutOfPlaneSite_getWeight13(vs)*v2 + _openmm.OutOfPlaneSite_getWeightCross(vs)*cross
                pos[i] = vspos
        newpos = pos*nanometer
        return newpos
    else: return positions

def CopyAmoebaBondParameters(src,dest):
    dest.setAmoebaGlobalBondCubic(src.getAmoebaGlobalBondCubic())
    dest.setAmoebaGlobalBondQuartic(src.getAmoebaGlobalBondQuartic())
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i,*src.getBondParameters(i))

def CopyAmoebaOutOfPlaneBendParameters(src,dest):
    dest.setAmoebaGlobalOutOfPlaneBendCubic(src.getAmoebaGlobalOutOfPlaneBendCubic())
    dest.setAmoebaGlobalOutOfPlaneBendQuartic(src.getAmoebaGlobalOutOfPlaneBendQuartic())
    dest.setAmoebaGlobalOutOfPlaneBendPentic(src.getAmoebaGlobalOutOfPlaneBendPentic())
    dest.setAmoebaGlobalOutOfPlaneBendSextic(src.getAmoebaGlobalOutOfPlaneBendSextic())
    for i in range(src.getNumOutOfPlaneBends()):
        dest.setOutOfPlaneBendParameters(i,*src.getOutOfPlaneBendParameters(i))

def CopyAmoebaAngleParameters(src, dest):
    dest.setAmoebaGlobalAngleCubic(src.getAmoebaGlobalAngleCubic())
    dest.setAmoebaGlobalAngleQuartic(src.getAmoebaGlobalAngleQuartic())
    dest.setAmoebaGlobalAnglePentic(src.getAmoebaGlobalAnglePentic())
    dest.setAmoebaGlobalAngleSextic(src.getAmoebaGlobalAngleSextic())
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i,*src.getAngleParameters(i))
    return

def CopyAmoebaInPlaneAngleParameters(src, dest):
    dest.setAmoebaGlobalInPlaneAngleCubic(src.getAmoebaGlobalInPlaneAngleCubic())
    dest.setAmoebaGlobalInPlaneAngleQuartic(src.getAmoebaGlobalInPlaneAngleQuartic())
    dest.setAmoebaGlobalInPlaneAnglePentic(src.getAmoebaGlobalInPlaneAnglePentic())
    dest.setAmoebaGlobalInPlaneAngleSextic(src.getAmoebaGlobalInPlaneAngleSextic())
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i,*src.getAngleParameters(i))
    return

def CopyAmoebaVdwParameters(src, dest):
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))

def CopyAmoebaMultipoleParameters(src, dest):
    for i in range(src.getNumMultipoles()):
        dest.setMultipoleParameters(i,*src.getMultipoleParameters(i))
    
def CopyHarmonicBondParameters(src, dest):
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i,*src.getBondParameters(i))

def CopyHarmonicAngleParameters(src, dest):
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i,*src.getAngleParameters(i))

def CopyPeriodicTorsionParameters(src, dest):
    for i in range(src.getNumTorsions()):
        dest.setTorsionParameters(i,*src.getTorsionParameters(i))

def CopyNonbondedParameters(src, dest):
    dest.setReactionFieldDielectric(src.getReactionFieldDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))
    for i in range(src.getNumExceptions()):
        dest.setExceptionParameters(i,*src.getExceptionParameters(i))

def do_nothing(src, dest):
    return

def CopySystemParameters(src,dest):
    """Copy parameters from one system (i.e. that which is created by a new force field)
    sto another system (i.e. the one stored inside the Target object).
    DANGER: These need to be implemented manually!!!"""
    Copiers = {'AmoebaBondForce':CopyAmoebaBondParameters,
               'AmoebaOutOfPlaneBendForce':CopyAmoebaOutOfPlaneBendParameters,
               'AmoebaAngleForce':CopyAmoebaAngleParameters,
               'AmoebaInPlaneAngleForce':CopyAmoebaInPlaneAngleParameters,
               'AmoebaVdwForce':CopyAmoebaVdwParameters,
               'AmoebaMultipoleForce':CopyAmoebaMultipoleParameters,
               'HarmonicBondForce':CopyHarmonicBondParameters,
               'HarmonicAngleForce':CopyHarmonicAngleParameters,
               'PeriodicTorsionForce':CopyPeriodicTorsionParameters,
               'NonbondedForce':CopyNonbondedParameters,
               'CMMotionRemover':do_nothing}
    for i in range(src.getNumForces()):
        nm = src.getForce(i).__class__.__name__
        if nm in Copiers:
            Copiers[nm](src.getForce(i),dest.getForce(i))
        else:
            warn_press_key('There is no Copier function implemented for the OpenMM force type %s!' % nm)

def UpdateSimulationParameters(src_system, dest_simulation):
    CopySystemParameters(src_system, dest_simulation.system)
    for i in range(src_system.getNumForces()):
        if hasattr(dest_simulation.system.getForce(i),'updateParametersInContext'):
            dest_simulation.system.getForce(i).updateParametersInContext(dest_simulation.context)

def SetAmoebaVirtualExclusions(system):
    if any([f.__class__.__name__ == "AmoebaMultipoleForce" for f in system.getForces()]):
        # print "Cajoling AMOEBA covalent maps so they work with virtual sites."
        vss = [(i, [system.getVirtualSite(i).getParticle(j) for j in range(system.getVirtualSite(i).getNumParticles())]) \
                   for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
        for f in system.getForces():
            if f.__class__.__name__ == "AmoebaMultipoleForce":
                # print "--- Before ---"
                # for i in range(f.getNumMultipoles()):
                #     print f.getCovalentMaps(i)
                for i, j in vss:
                    f.setCovalentMap(i, 0, j)
                    f.setCovalentMap(i, 4, j+[i])
                    for k in j:
                        f.setCovalentMap(k, 0, list(f.getCovalentMap(k, 0))+[i])
                        f.setCovalentMap(k, 4, list(f.getCovalentMap(k, 4))+[i])
                # print "--- After ---"
                # for i in range(f.getNumMultipoles()):
                #     print f.getCovalentMaps(i)

def MTSVVVRIntegrator(temperature, collision_rate, timestep, system, ninnersteps=4):
    """
    Create a multiple timestep velocity verlet with velocity randomization (VVVR) integrator.
    
    ARGUMENTS

    temperature (numpy.unit.Quantity compatible with kelvin) - the temperature
    collision_rate (numpy.unit.Quantity compatible with 1/picoseconds) - the collision rate
    timestep (numpy.unit.Quantity compatible with femtoseconds) - the integration timestep
    system (simtk.openmm.System) - system whose forces will be partitioned
    ninnersteps (int) - number of inner timesteps (default: 4)

    RETURNS

    integrator (openmm.CustomIntegrator) - a VVVR integrator

    NOTES
    
    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    timestep correction to ensure that the field-free diffusion constant is timestep invariant.  The inner
    velocity Verlet discretization is transformed into a multiple timestep algorithm.

    REFERENCES

    VVVR Langevin integrator: 
    * http://arxiv.org/abs/1301.3800
    * http://arxiv.org/abs/1107.2967 (to appear in PRX 2013)    
    
    TODO

    Move initialization of 'sigma' to setting the per-particle variables.
    
    """
    # Multiple timestep Langevin integrator.
    for i in system.getForces():
        if i.__class__.__name__ in ["NonbondedForce", "CustomNonbondedForce", "AmoebaVdwForce", "AmoebaMultipoleForce"]:
            # Slow force.
            logger.info(i.__class__.__name__ + "is a Slow Force\n")
            i.setForceGroup(1)
        else:
            logger.info(i.__class__.__name__ + "is a Fast Force\n")
            # Fast force.
            i.setForceGroup(0)

    kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    
    integrator = CustomIntegrator(timestep)
    
    integrator.addGlobalVariable("dt_fast", timestep/float(ninnersteps)) # fast inner timestep
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("a", numpy.exp(-collision_rate*timestep)) # velocity mixing parameter
    integrator.addGlobalVariable("b", numpy.sqrt((2/(collision_rate*timestep)) * numpy.tanh(collision_rate*timestep/2))) # timestep correction parameter
    integrator.addPerDofVariable("sigma", 0) 
    integrator.addPerDofVariable("x1", 0) # position before application of constraints

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    # 
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(a)*v + sqrt(1-a)*sigma*gaussian")
    integrator.addConstrainVelocities();
    
    #
    # Symplectic inner multiple timestep.
    #
    integrator.addUpdateContextState(); 
    integrator.addComputePerDof("v", "v + 0.5*b*dt*f1/m")
    for innerstep in range(ninnersteps):
        # Fast inner symplectic timestep.
        integrator.addComputePerDof("v", "v + 0.5*b*dt_fast*f0/m")
        integrator.addComputePerDof("x", "x + v*b*dt_fast")
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainPositions();        
        integrator.addComputePerDof("v", "v + 0.5*b*dt_fast*f0/m + (x-x1)/dt_fast")
    integrator.addComputePerDof("v", "v + 0.5*b*dt*f1/m") # TODO: Additional velocity constraint correction?
    integrator.addConstrainVelocities();

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(a)*v + sqrt(1-a)*sigma*gaussian")
    integrator.addConstrainVelocities();

    return integrator


"""Dictionary for building parameter identifiers.  As usual they go like this:
Bond/length/OW.HW
The dictionary is two-layered because the same interaction type (Bond)
could be under two different parent types (HarmonicBondForce, AmoebaHarmonicBondForce)"""
suffix_dict = { "HarmonicBondForce" : {"Bond" : ["class1","class2"]},
                "HarmonicAngleForce" : {"Angle" : ["class1","class2","class3"],},
                "PeriodicTorsionForce" : {"Proper" : ["class1","class2","class3","class4"],},
                "NonbondedForce" : {"Atom": ["type"]},
                "AmoebaBondForce" : {"Bond" : ["class1","class2"]},
                "AmoebaAngleForce" : {"Angle" : ["class1","class2","class3"]},
                "AmoebaStretchBendForce" : {"StretchBend" : ["class1","class2","class3"]},
                "AmoebaVdwForce" : {"Vdw" : ["class"]},
                "AmoebaMultipoleForce" : {"Multipole" : ["type","kz","kx"], "Polarize" : ["type"]},
                "AmoebaUreyBradleyForce" : {"UreyBradley" : ["class1","class2","class3"]},
                "Residues.Residue" : {"VirtualSite" : ["index"]}
                }

## pdict is a useless variable if the force field is XML.
pdict = "XML_Override"

class OpenMM_Reader(BaseReader):
    """ Class for parsing OpenMM force field files. """
    def __init__(self,fnm):
        ## Initialize the superclass. :)
        super(OpenMM_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict

    def build_pid(self, element, parameter):
        """ Build the parameter identifier (see _link_ for an example)
        @todo Add a link here """
        #InteractionType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:] + [element.tag])
        ParentType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:])
        InteractionType = element.tag
        try:
            if ParentType == "Residues.Residue":
                pfx = list(element.iterancestors())[0].attrib["name"]
                Involved = '.'.join([pfx+"-"+element.attrib[i] for i in suffix_dict[ParentType][InteractionType]])
            else:
                Involved = '.'.join([element.attrib[i] for i in suffix_dict[ParentType][InteractionType] if i in element.attrib])
            return "/".join([InteractionType, parameter, Involved])
        except:
            logger.info("Minor warning: Parameter ID %s doesn't contain any atom types, redundancies are possible\n" % ("/".join([InteractionType, parameter])))
            return "/".join([InteractionType, parameter])

class Liquid_OpenMM(Liquid):
    def __init__(self,options,tgt_opts,forcefield):
        super(Liquid_OpenMM,self).__init__(options,tgt_opts,forcefield)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'force_cuda',forceprint=True)
        # Enable multiple timestep integrator
        self.set_option(tgt_opts,'mts_vvvr',forceprint=True)
        # Set up for polarization correction
        if self.do_self_pol:
            self.mpdb = PDBFile(os.path.join(self.root,self.tgtdir,"gas.pdb"))
            forcefield = ForceField(os.path.join(self.root,options['ffdir'],self.FF.openmmxml))
            mod = Modeller(self.mpdb.topology, self.mpdb.positions)
            mod.addExtraParticles(forcefield)
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water)
            SetAmoebaVirtualExclusions(system)
            # Create the simulation; we're not actually going to use the integrator
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            # Create a Reference platform (this will be faster than CUDA since it's small)
            self.platform = Platform.getPlatformByName('Reference')
            # Create the simulation object
            self.msim = Simulation(mod.topology, system, integrator, self.platform)
        # I shall enable starting simulations for many different initial conditions.
        self.liquid_fnm = "liquid.pdb"
        self.liquid_conf = Molecule(os.path.join(self.root, self.tgtdir,"liquid.pdb"))
        self.liquid_mol = None
        self.gas_fnm = "gas.pdb"
        if os.path.exists(os.path.join(self.root, self.tgtdir,"all.gro")):
            self.liquid_mol = Molecule(os.path.join(self.root, self.tgtdir,"all.gro"))
            logger.info("Found collection of starting conformations, length %i!\n" % len(self.liquid_mol))
        # Prefix to command string for launching NPT simulations.
        self.nptpfx += "bash runcuda.sh"
        # List of extra files to upload to Work Queue.
        self.nptfiles += ['runcuda.sh']
        # Suffix to command string for launching NPT simulations.
        self.nptsfx += [("--force_cuda" if self.force_cuda else None),
                        ("--anisotropic" if self.anisotropic_box else None),
                        ("--mts_vvvr" if self.mts_vvvr else None)]
        # MD engine argument supplied to command string for launching NPT simulations.
        self.engine = "openmm"

    def prepare_temp_directory(self,options,tgt_opts):
        """ Prepare the temporary directory by copying in important files. """
        abstempdir = os.path.join(self.root,self.tempdir)
        # The liquid.pdb file is not written here.
        LinkFile(os.path.join(self.root,self.tgtdir,"gas.pdb"),os.path.join(abstempdir,"gas.pdb"))
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","runcuda.sh"),os.path.join(abstempdir,"runcuda.sh"))
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","npt.py"),os.path.join(abstempdir,"npt.py"))

    def polarization_correction(self,mvals):
        self.FF.make(mvals)
        ff = ForceField(self.FF.openmmxml)
        mod = Modeller(self.mpdb.topology, self.mpdb.positions)
        mod.addExtraParticles(ff)
        system = ff.createSystem(mod.topology, rigidWater=self.FF.rigid_water)
        # SetAmoebaVirtualExclusions(system)
        UpdateSimulationParameters(system, self.msim)
        self.msim.context.setPositions(mod.getPositions())
        self.msim.minimizeEnergy()
        pos = self.msim.context.getState(getPositions=True).getPositions()
        pos = ResetVirtualSites(pos, system)
        d = get_dipole(self.msim, positions=pos)
        if not in_fd():
            logger.info("The molecular dipole moment is % .3f debye\n" % np.linalg.norm(d))
        dd2 = ((np.linalg.norm(d)-self.self_pol_mu0)*debye)**2
        eps0 = 8.854187817620e-12 * coulomb**2 / newton / meter**2
        epol = 0.5*dd2/(self.self_pol_alpha*angstrom**3*4*np.pi*eps0)/(kilojoule_per_mole/AVOGADRO_CONSTANT_NA)
        return epol

class OpenMM(Engine):

    """ Derived from Engine object for carrying out general purpose OpenMM calculations. """

    def __init__(self, name="openmm", **kwargs):
        self.valkwd = ['ffxml', 'pdb', 'platname', 'precision', 'mmopts']
        super(OpenMM,self).__init__(name=name, **kwargs)

    def setopts(self, platname="CUDA", precision="single", **kwargs):

        """ Called by __init__ ; Set OpenMM-specific options. """

        ## Target settings override.
        if hasattr(self,'target'):
            self.platname = self.target.platname
            self.precision = self.target.precision
        else:
            self.platname = platname
            self.precision = precision

        valnames = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
        if self.platname not in valnames:
            warn_press_key("Platform %s does not exist (valid options are %s (case-sensitive))" % (self.platname, valnames))
            self.platname = 'Reference'
        self.precision = self.precision.lower()
        valprecs = ['single','mixed','double']
        if self.precision not in valprecs:
            raise RuntimeError("Please specify one of %s for precision" % valprecs)
        ## Set the simulation platform
        logger.info("Setting Platform to %s\n" % self.platname)
        self.platform = Platform.getPlatformByName(self.platname)
        if self.platname == 'CUDA':
            ## Set the device to the environment variable or zero otherwise
            device = os.environ.get('CUDA_DEVICE',"0")
            logger.info("Setting CUDA Device to %s\n" % device)
            self.platform.setPropertyDefaultValue("CudaDeviceIndex", device)
            logger.info("Setting CUDA Precision to %s\n" % self.precision)
            self.platform.setPropertyDefaultValue("CudaPrecision", self.precision)
        elif self.platname == 'OPENCL':
            device = os.environ.get('OPENCL_DEVICE',"0")
            logger.info("Setting OpenCL Device to %s\n" % device)
            self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
            logger.info("Setting OpenCL Precision to %s\n" % self.precision)
            self.platform.setPropertyDefaultValue("OpenCLPrecision", self.precision)

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory.  
        Provide a molecule object or a coordinate file.
        Add an optional PDB file for residues, atom names etc. """

        pdbfnm = None
        # Determine the PDB file name.
        if 'pdb' in kwargs and os.path.exists(kwargs['pdb']):
            # Case 1. The PDB file name is provided explicitly
            pdbfnm = kwargs['pdb']
            if not os.path.exists(pdbfnm): raise RuntimeError("%s specified but doesn't exist" % pdbfnm)

        if 'mol' in kwargs:
            self.mol = kwargs['mol']
        elif 'coords' in kwargs:
            self.mol = Molecule(kwargs['coords'])
        else:
            raise RuntimeError('Must provide either a molecule object or coordinate file.')

        if pdbfnm != None:
            mpdb = Molecule(pdbfnm)
            for i in ["chain", "atomname", "resid", "resname", "elem"]:
                self.mol.Data[i] = mpdb.Data[i]

    def prepare(self, mmopts={}, **kwargs):

        """ Prepare the temp-directory. """

        ## OpenMM needs to create a PDB object.
        pdb1 = "%s-1.pdb" % os.path.splitext(self.mol.fnm)[0]
        self.mol[0].write(pdb1)
        self.pdb = PDBFile(pdb1)

        if hasattr(self, 'target'):
            FF = self.target.FF
            self.ffxml = FF.openmmxml
            forcefield = ForceField(os.path.join(self.root, FF.ffdir, FF.openmmxml))
        else:
            if 'ffxml' in kwargs:
                if not os.path.exists(kwargs['ffxml']): 
                    raise RuntimeError("%s doesn't exist" % kwargs['ffxml'])
                self.ffxml = kwargs['ffxml']
            elif onefile('xml'):
                self.ffxml = onefile('xml')
            forcefield = ForceField(self.ffxml)
            
        ## Create the simulation object within this class itself.
        mod = Modeller(self.pdb.topology, self.pdb.positions)
        mod.addExtraParticles(forcefield)

        self.mmopts = mmopts
        if hasattr(self,'target'):
            FF = self.target.FF
            if FF.amoeba_pol == 'mutual':
                self.mmopts['mutualInducedTargetEpsilon'] = 1e-6
            elif FF.amoeba_pol == 'direct':
                self.mmopts['polarization'] = 'Direct'
            self.mmopts['rigidWater'] = FF.rigid_water

        system = forcefield.createSystem(mod.topology, **self.mmopts)
        # Set up for energy component analysis.
        for i, j in enumerate(system.getForces()):
            j.setForceGroup(i)
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        SetAmoebaVirtualExclusions(system)
        self.simulation = Simulation(mod.topology, system, integrator, self.platform)
        # Generate OpenMM-compatible positions
        self.xyz_omms = []
        for I in range(len(self.mol)):
            xyz = self.mol.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # An extra step with adding virtual particles
            mod = Modeller(self.pdb.topology, xyz_omm)
            mod.addExtraParticles(forcefield)
            # Set the positions using the trajectory
            self.xyz_omms.append(mod.getPositions())

        # This could go into prepare(), but it really doesn't depend on which directory we're in.
        Top = self.pdb.getTopology()
        Atoms = list(Top.atoms())
        Bonds = [(a.index, b.index) for a, b in list(Top.bonds())]
        self.AtomMask = []
        self.AtomLists = defaultdict(list)
        self.AtomLists['Mass'] = [a.element.mass.value_in_unit(dalton) if a.element != None else 0 for a in Atoms]
        self.AtomLists['ParticleType'] = ['A' if m >= 1.0 else 'D' for m in self.AtomLists['Mass']]
        self.AtomLists['ResidueNumber'] = [a.residue.index for a in Atoms]
        G = nx.Graph()
        for a in Atoms:
            G.add_node(a.index)
        for a, b in Bonds:
            G.add_edge(a, b)
        # Use networkx to figure out a list of molecule numbers.
        gs = nx.connected_component_subgraphs(G)
        tmols = [gs[i] for i in np.argsort(np.array([min(g.nodes()) for g in gs]))]
        self.AtomLists['MoleculeNumber'] = [[i in m.nodes() for m in tmols].index(1) for i in range(self.mol.na)]
        self.AtomMask = [a == 'A' for a in self.AtomLists['ParticleType']]

    def update_simulation(self):
        """ Update the force field parameters in the simulation object. """
        forcefield = ForceField(self.ffxml)
        mod = Modeller(self.pdb.topology, self.pdb.positions)
        mod.addExtraParticles(forcefield)
        self.system = forcefield.createSystem(mod.topology, **self.mmopts)
        UpdateSimulationParameters(self.system, self.simulation)

    def set_positions(self, shot=0):
        self.simulation.context.setPositions(ResetVirtualSites(self.xyz_omms[shot], self.system))

    def energy_force(self, force=True):
        """ Loop through the snapshots and compute the energies and forces using OpenMM. """

        self.update_simulation()
        M = []
        # Loop through the snapshots
        for I in range(len(self.mol)):
            # Ideally the virtual site parameters would be copied but they're not.
            # Instead we update the vsite positions manually.
            # if self.FF.rigid_water:
            #     simulation.context.applyConstraints(1e-8)
            # else:
            #     simulation.context.computeVirtualSites()
            self.set_positions(I)
            Energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
            Force1 = []
            if force:
                # Compute the force and append to list
                Force = list(np.array(self.simulation.context.getState(getForces=True).getForces() / kilojoules_per_mole * nanometer).flatten())
                # Extract forces belonging to real atoms only
                Force1 = list(itertools.chain(*[Force[3*i:3*i+3] for i in range(len(Force)/3) if self.AtomMask[i]]))
            M.append(np.array([Energy] + Force1))
        M = np.array(M)
        return M

    def energy(self):
        return self.energy_force(force=False).flatten()

    def normal_modes(self, optimize=True):
        raise NotImplementedError

    def multipole_moments(self, optimize=True, polarizability=False):

        """ Return the system multipole moments, optionally optimizing the geometry first. """

        system = self.update_simulation()
        self.set_positions()

        if polarizability:
            raise NotImplementedError

        rmsd = 0.0

        if optimize:
            self.simulation.minimizeEnergy(tolerance=1e-4*kilojoule/mole)
        moments = get_multipoles(self.simulation)
        
        dipole_dict = OrderedDict(zip(['x','y','z'], moments[:3]))
        quadrupole_dict = OrderedDict(zip(['xx','xy','yy','xz','yz','zz'], moments[3:10]))

        calc_moments = OrderedDict([('dipole', dipole_dict), ('quadrupole', quadrupole_dict)])

        return calc_moments

    def energy_rmsd(self, optimize=True):

        """ Calculate energy of the starting structure, with an optional energy minimization and RMSD calculation. """

        system = self.update_simulation()

        rmsd = 0.0
        X0 = self.xyz_omms[0]
        self.simulation.context.setPositions(X0)
        if optimize:
            self.simulation.minimizeEnergy(tolerance=1e-4*kilojoule/mole)
            S = self.simulation.context.getState(getPositions=True, getEnergy=True)
            E = S.getPotentialEnergy().value_in_unit(kilocalories_per_mole)
            M0 = deepcopy(self.mol)
            X1 = S.getPositions().value_in_unit(angstrom)
            M0.xyzs.append(np.array([j for i, j in enumerate(X1) if self.AtomMask[i]]))
            rmsd = M0.ref_rmsd(0)[1]
        else:
            E = self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)
        return E, rmsd

    def interaction_energy(self, fraga, fragb):
        
        """ Calculate the interaction energy for two fragments. """

        if self.name == 'A' or self.name == 'B':
            raise RuntimeError("Don't name the engine A or B!")

        # Create two subengines.
        if hasattr(self,'target'):
            if not hasattr(self,'A'):
                self.A = OpenMM(name="A", mol=self.mol.atom_select(fraga), target=self.target)
            if not hasattr(self,'B'):
                self.B = OpenMM(name="B", mol=self.mol.atom_select(fragb), target=self.target)
        else:
            if not hasattr(self,'A'):
                self.A = OpenMM(name="A", mol=self.mol.atom_select(fraga), platname=self.platname, \
                                    precision=self.precision, ffxml=self.ffxml, mmopts=self.mmopts)
            if not hasattr(self,'B'):
                self.B = OpenMM(name="B", mol=self.mol.atom_select(fragb), platname=self.platname, \
                                    precision=self.precision, ffxml=self.ffxml, mmopts=self.mmopts)

        # Interaction energy needs to be in kcal/mol.
        D = self.energy() 
        A = self.A.energy()
        B = self.B.energy()

        return (D - A - B) / 4.184

class AbInitio_OpenMM(AbInitio):

    """Subclass of AbInitio for force and energy matching
    using OpenMM.  Implements the prepare and energy_force_driver
    methods.  The get method is in the superclass.  """

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'pdb',default="conf.pdb")
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double")
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA")
        ## Initialize base class.
        super(AbInitio_OpenMM,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
        ## Create engine object.
        self.engine = OpenMM(target=self, **engine_args)
        self.AtomLists = self.engine.AtomLists
        self.AtomMask = self.engine.AtomMask

    def energy_force_driver_all(self):
        return self.engine.energy_force()

class Interaction_OpenMM(Interaction):

    """Subclass of Target for interaction matching using OpenMM. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="all.pdb")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double")
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA")
        ## Initialize base class.
        super(Interaction_OpenMM,self).__init__(options,tgt_opts,forcefield)
        ## Build keyword dictionaries to pass to engine.
        engine_args = deepcopy(self.__dict__)
        engine_args.update(options)
        del engine_args['name']
        ## Create engine object.
        self.engine = OpenMM(target=self, **engine_args)

    def interaction_driver_all(self,dielectric=False):
        return self.engine.interaction_energy(self.select1, self.select2)

