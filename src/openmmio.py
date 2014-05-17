""" @package forcebalance.openmmio OpenMM input/output.

@author Lee-Ping Wang
@date 04/2012
"""

import os
from forcebalance import BaseReader
from forcebalance.abinitio import AbInitio
from forcebalance.binding import BindingEnergy
from forcebalance.liquid import Liquid
from forcebalance.interaction import Interaction
from forcebalance.moments import Moments
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
    if type(Sim.integrator) in [LangevinIntegrator, VerletIntegrator]:
        for i in range(Sim.system.getNumForces()):
            EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
    return EnergyTerms

def get_multipoles(simulation,q=None,mass=None,positions=None,rmcom=True):
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
        if isinstance(i, AmoebaMultipoleForce):
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
        if isinstance(i, NonbondedForce):
            # Get array of charges.
            if q == None:
                q = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])
            # Get array of positions in nanometers.
            if positions == None:
                positions = simulation.context.getState(getPositions=True).getPositions()
            if mass == None:
                mass = np.array([simulation.context.getSystem().getParticleMass(k).value_in_unit(dalton) \
                                     for k in range(simulation.context.getSystem().getNumParticles())])
            x = np.array(positions.value_in_unit(nanometer))
            if rmcom:
                com = np.sum(x*mass.reshape(-1,1),axis=0) / np.sum(mass)
                x -= com
            xx, xy, xz, yy, yz, zz = (x[:,k]*x[:,l] for k, l in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)])
            # Multiply charges by positions to get dipole moment.
            dip = enm_debye * np.sum(x*q.reshape(-1,1),axis=0)
            dx += dip[0]
            dy += dip[1]
            dz += dip[2]
            qxx += enm_debye * 15 * np.sum(q*xx)
            qxy += enm_debye * 15 * np.sum(q*xy)
            qxz += enm_debye * 15 * np.sum(q*xz)
            qyy += enm_debye * 15 * np.sum(q*yy)
            qyz += enm_debye * 15 * np.sum(q*yz)
            qzz += enm_debye * 15 * np.sum(q*zz)
            tr = qxx+qyy+qzz
            qxx -= tr/3
            qyy -= tr/3
            qzz -= tr/3
    # This ordering has to do with the way TINKER prints it out.
    return [dx,dy,dz,qxx,qxy,qyy,qxz,qyz,qzz]

def get_dipole(simulation,q=None,mass=None,positions=None):
    """Return the current dipole moment in Debye.
    Note that this quantity is meaningless if the system carries a net charge."""
    return get_multipoles(simulation, q=q, mass=mass, positions=positions, rmcom=False)[:3]

def PrepareVirtualSites(system):
    """ Prepare a list of function wrappers and vsite parameters from the system. """
    isvsites = []
    vsfuncs = []
    vsidxs = []
    vswts = []
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            isvsites.append(1)
            vs = system.getVirtualSite(i)
            if isinstance(vs, TwoParticleAverageSite):
                vsidx = [_openmm.VirtualSite_getParticle(vs, 0), _openmm.VirtualSite_getParticle(vs, 1)]
                vswt = [_openmm.TwoParticleAverageSite_getWeight(vs, 0), _openmm.TwoParticleAverageSite_getWeight(vs, 1)]
                def vsfunc(pos, idx_, wt_):
                    return wt_[0]*pos[idx_[0]] + wt_[1]*pos[idx_[1]]
            elif isinstance(vs, ThreeParticleAverageSite):
                vsidx = [_openmm.VirtualSite_getParticle(vs, 0), _openmm.VirtualSite_getParticle(vs, 1), _openmm.VirtualSite_getParticle(vs, 2)]
                vswt = [_openmm.ThreeParticleAverageSite_getWeight(vs, 0), _openmm.ThreeParticleAverageSite_getWeight(vs, 1), _openmm.ThreeParticleAverageSite_getWeight(vs, 2)]
                def vsfunc(pos, idx_, wt_):
                    return wt_[0]*pos[idx_[0]] + wt_[1]*pos[idx_[1]] + wt_[2]*pos[idx_[2]]
            elif isinstance(vs, OutOfPlaneSite):
                vsidx = [_openmm.VirtualSite_getParticle(vs, 0), _openmm.VirtualSite_getParticle(vs, 1), _openmm.VirtualSite_getParticle(vs, 2)]
                vswt = [_openmm.OutOfPlaneSite_getWeight12(vs), _openmm.OutOfPlaneSite_getWeight13(vs), _openmm.OutOfPlaneSite_getWeightCross(vs)]
                def vsfunc(pos, idx_, wt_):
                    v1 = pos[idx_[1]] - pos[idx_[0]]
                    v2 = pos[idx_[2]] - pos[idx_[0]]
                    cross = np.array([v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0]])
                    return pos[idx_[0]] + wt_[0]*v1 + wt_[1]*v2 + wt_[2]*cross
        else:
            isvsites.append(0)
            vsfunc = None
            vsidx = None
            vswt = None
        vsfuncs.append(deepcopy(vsfunc))
        vsidxs.append(deepcopy(vsidx))
        vswts.append(deepcopy(vswt))
    return (isvsites, vsfuncs, vsidxs, vswts)

def ResetVirtualSites_fast(positions, vsinfo):
    """Given a set of OpenMM-compatible positions and a System object,
    compute the correct virtual site positions according to the System."""
    isvsites, vsfuncs, vsidxs, vswts = vsinfo
    if any(isvsites):
        pos = np.array(positions.value_in_unit(nanometer))
        for i in range(len(positions)):
            if isvsites[i]:
                pos[i] = vsfuncs[i](pos, vsidxs[i], vswts[i])
        newpos = [Vec3(*i) for i in pos]*nanometer
        return newpos
    else:
        return positions

def ResetVirtualSites(positions, system):
    """Given a set of OpenMM-compatible positions and a System object,
    compute the correct virtual site positions according to the System."""
    if any([system.isVirtualSite(j) for j in range(system.getNumParticles())]):
        pos = np.array(positions.value_in_unit(nanometer))
        for i in range(system.getNumParticles()):
            if system.isVirtualSite(i):
                vs = system.getVirtualSite(i)
                # Faster code to work around Python API slowness.
                if isinstance(vs, TwoParticleAverageSite):
                    vspos = _openmm.TwoParticleAverageSite_getWeight(vs, 0)*pos[_openmm.VirtualSite_getParticle(vs, 0)] \
                        + _openmm.TwoParticleAverageSite_getWeight(vs, 1)*pos[_openmm.VirtualSite_getParticle(vs, 1)]
                elif isinstance(vs, ThreeParticleAverageSite):
                    vspos = _openmm.ThreeParticleAverageSite_getWeight(vs, 0)*pos[_openmm.VirtualSite_getParticle(vs, 0)] \
                        + _openmm.ThreeParticleAverageSite_getWeight(vs, 1)*pos[_openmm.VirtualSite_getParticle(vs, 1)] \
                        + _openmm.ThreeParticleAverageSite_getWeight(vs, 2)*pos[_openmm.VirtualSite_getParticle(vs, 2)]
                elif isinstance(vs, OutOfPlaneSite):
                    v1 = pos[_openmm.VirtualSite_getParticle(vs, 1)] - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    v2 = pos[_openmm.VirtualSite_getParticle(vs, 2)] - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    cross = Vec3(v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0])
                    vspos = pos[_openmm.VirtualSite_getParticle(vs, 0)] + _openmm.OutOfPlaneSite_getWeight12(vs)*v1 \
                        + _openmm.OutOfPlaneSite_getWeight13(vs)*v2 + _openmm.OutOfPlaneSite_getWeightCross(vs)*cross
                pos[i] = vspos
        newpos = [Vec3(*i) for i in pos]*nanometer
        return newpos
    else: return positions

def GetVirtualSiteParameters(system):
    """Return an array of all virtual site parameters in the system.  Used for comparison purposes."""
    vsprm = []
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            vs = system.getVirtualSite(i)
            vstype = vs.__class__.__name__
            if vstype == 'TwoParticleAverageSite':
                vsprm.append(_openmm.TwoParticleAverageSite_getWeight(vs, 0))
                vsprm.append(_openmm.TwoParticleAverageSite_getWeight(vs, 1))
            elif vstype == 'ThreeParticleAverageSite':
                vsprm.append(_openmm.ThreeParticleAverageSite_getWeight(vs, 0))
                vsprm.append(_openmm.ThreeParticleAverageSite_getWeight(vs, 1))
                vsprm.append(_openmm.ThreeParticleAverageSite_getWeight(vs, 2))
            elif vstype == 'OutOfPlaneSite':
                vsprm.append(_openmm.OutOfPlaneSite_getWeight12(vs))
                vsprm.append(_openmm.OutOfPlaneSite_getWeight13(vs))
                vsprm.append(_openmm.OutOfPlaneSite_getWeightCross(vs))
    return np.array(vsprm)

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
        # logger.info("Cajoling AMOEBA covalent maps so they work with virtual sites.\n")
        vss = [(i, [system.getVirtualSite(i).getParticle(j) for j in range(system.getVirtualSite(i).getNumParticles())]) \
                   for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
        for f in system.getForces():
            if f.__class__.__name__ == "AmoebaMultipoleForce":
                # logger.info("--- Before ---\n")
                # for i in range(f.getNumMultipoles()):
                #     logger.info("%s\n" % f.getCovalentMaps(i))
                for i, j in vss:
                    f.setCovalentMap(i, 0, j)
                    f.setCovalentMap(i, 4, j+[i])
                    for k in j:
                        f.setCovalentMap(k, 0, list(f.getCovalentMap(k, 0))+[i])
                        f.setCovalentMap(k, 4, list(f.getCovalentMap(k, 4))+[i])
                # logger.info("--- After ---\n")
                # for i in range(f.getNumMultipoles()):
                #     logger.info("%s\n" % f.getCovalentMaps(i))

def MTSVVVRIntegrator(temperature, collision_rate, timestep, system, ninnersteps=4):
    """
    Create a multiple timestep velocity verlet with velocity randomization (VVVR) integrator.
    
    ARGUMENTS

    temperature (Quantity compatible with kelvin) - the temperature
    collision_rate (Quantity compatible with 1/picoseconds) - the collision rate
    timestep (Quantity compatible with femtoseconds) - the integration timestep
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
            logger.info(i.__class__.__name__ + " is a Slow Force\n")
            i.setForceGroup(1)
        else:
            logger.info(i.__class__.__name__ + " is a Fast Force\n")
            # Fast force.
            i.setForceGroup(0)

    kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
    kT = kB * temperature
    
    integrator = CustomIntegrator(timestep)
    
    integrator.addGlobalVariable("dt_fast", timestep/float(ninnersteps)) # fast inner timestep
    integrator.addGlobalVariable("kT", kT) # thermal energy
    integrator.addGlobalVariable("a", np.exp(-collision_rate*timestep)) # velocity mixing parameter
    integrator.addGlobalVariable("b", np.sqrt((2/(collision_rate*timestep)) * np.tanh(collision_rate*timestep/2))) # timestep correction parameter
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
            logger.error("Please specify one of %s for precision\n" % valprecs)
            raise RuntimeError
        ## Set the simulation platform
        if self.verbose: logger.info("Setting Platform to %s\n" % self.platname)
        self.platform = Platform.getPlatformByName(self.platname)
        if self.platname == 'CUDA':
            ## Set the device to the environment variable or zero otherwise
            device = os.environ.get('CUDA_DEVICE',"0")
            if self.verbose: logger.info("Setting CUDA Device to %s\n" % device)
            self.platform.setPropertyDefaultValue("CudaDeviceIndex", device)
            if self.verbose: logger.info("Setting CUDA Precision to %s\n" % self.precision)
            self.platform.setPropertyDefaultValue("CudaPrecision", self.precision)
        elif self.platname == 'OpenCL':
            device = os.environ.get('OPENCL_DEVICE',"0")
            if self.verbose: logger.info("Setting OpenCL Device to %s\n" % device)
            self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
            if self.verbose: logger.info("Setting OpenCL Precision to %s\n" % self.precision)
            self.platform.setPropertyDefaultValue("OpenCLPrecision", self.precision)
        self.simkwargs = {}

    def readsrc(self, **kwargs):

        """ Called by __init__ ; read files from the source directory.  
        Provide a molecule object or a coordinate file.
        Add an optional PDB file for residues, atom names etc. """

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
            self.mol = Molecule(kwargs['coords'])
        else:
            logger.error('Must provide either a molecule object or coordinate file.\n')
            raise RuntimeError

        if pdbfnm != None:
            mpdb = Molecule(pdbfnm)
            for i in ["chain", "atomname", "resid", "resname", "elem"]:
                self.mol.Data[i] = mpdb.Data[i]

    def prepare(self, pbc=False, mmopts={}, **kwargs):

        """ 
        Prepare the calculation.  Note that we don't create the
        Simulation object yet, because that may depend on MD
        integrator parameters, thermostat, barostat etc.
        """
        ## Create the OpenMM PDB object.
        pdb1 = "%s-1.pdb" % os.path.splitext(os.path.basename(self.mol.fnm))[0]
        self.mol[0].write(pdb1)
        self.pdb = PDBFile(pdb1)
        os.unlink(pdb1)
        
        ## Create the OpenMM ForceField object.

        if hasattr(self, 'FF'):
            self.ffxml = self.FF.openmmxml
            self.forcefield = ForceField(os.path.join(self.root, self.FF.ffdir, self.FF.openmmxml))
        else:
            if 'ffxml' in kwargs:
                if type(kwargs['ffxml']) == list:
                    for i in kwargs['ffxml']:
                        if not os.path.exists(i):
                            logger.error("%s doesn't exist\n" % i)
                            raise RuntimeError
                else:
                    if not os.path.exists(kwargs['ffxml']): 
                        logger.error("%s doesn't exist\n" % kwargs['ffxml'])
                        raise RuntimeError
                self.ffxml = kwargs['ffxml']
            elif onefile('xml'):
                self.ffxml = onefile('xml')
            if type(self.ffxml) == list:
                self.forcefield = ForceField(*self.ffxml)
            else:
                self.forcefield = ForceField(self.ffxml)

        ## OpenMM options for setting up the System.
        self.mmopts = dict(mmopts)

        ## Are we using AMOEBA?
        self.AMOEBA = any(['Amoeba' in f.__class__.__name__ for f in self.forcefield._forces])

        ## Set system options from ForceBalance force field options.
        if hasattr(self,'FF'):
            if self.AMOEBA:
                if self.FF.amoeba_pol == None:
                    logger.error('You must specify amoeba_pol if there are any AMOEBA forces.\n')
                    raise RuntimeError
                if self.FF.amoeba_pol == 'mutual':
                    self.mmopts['polarization'] = 'mutual'
                    self.mmopts.setdefault('mutualInducedTargetEpsilon', self.FF.amoeba_eps if self.FF.amoeba_eps != None else 1e-6)
                    self.mmopts['mutualInducedMaxIterations'] = 500
                elif self.FF.amoeba_pol == 'direct':
                    self.mmopts['polarization'] = 'direct'
            self.mmopts['rigidWater'] = self.FF.rigid_water

        ## Set system options from periodic boundary conditions.
        self.pbc = pbc
        if pbc:
            self.mmopts.setdefault('nonbondedMethod', PME)
            if self.AMOEBA:
                self.mmopts.setdefault('nonbondedCutoff', 0.7*nanometer)
                self.mmopts.setdefault('vdwCutoff', 0.85)
                self.mmopts.setdefault('aEwald', 5.4459052)
                self.mmopts.setdefault('pmeGridDimensions', [24,24,24])
            else:
                self.mmopts.setdefault('nonbondedCutoff', 0.85*nanometer)
                self.mmopts.setdefault('useSwitchingFunction', True)
                self.mmopts.setdefault('switchingDistance', 0.75*nanometer)
            self.mmopts.setdefault('useDispersionCorrection', True)
        else:
            self.mmopts.setdefault('nonbondedMethod', NoCutoff)
            self.mmopts['removeCMMotion'] = False

        ## Generate OpenMM-compatible positions
        self.xyz_omms = []
        for I in range(len(self.mol)):
            xyz = self.mol.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # An extra step with adding virtual particles
            mod = Modeller(self.pdb.topology, xyz_omm)
            mod.addExtraParticles(self.forcefield)
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
        self.AtomLists['Mass'] = [a.element.mass.value_in_unit(dalton) if a.element != None else 0 for a in Atoms]
        self.AtomLists['ParticleType'] = ['A' if m >= 1.0 else 'D' for m in self.AtomLists['Mass']]
        self.AtomLists['ResidueNumber'] = [a.residue.index for a in Atoms]
        self.AtomMask = [a == 'A' for a in self.AtomLists['ParticleType']]

    def create_simulation(self, timestep=1.0, faststep=0.25, temperature=None, pressure=None, anisotropic=False, mts=False, collision=1.0, nbarostat=25, rpmd_beads=0, **kwargs):

        """
        Create simulation object.  Note that this also takes in some
        options pertinent to system setup, including the type of MD
        integrator and type of pressure control.
        """

        # Divisor for the temperature (RPMD sets it to nonzero.)
        self.tdiv = 1

        ## Determine the integrator.
        if temperature:
            ## If temperature control is turned on, then run Langevin dynamics.
            if mts:
                if rpmd_beads > 0:
                    logger.error("No multiple timestep integrator without temperature control.\n")
                    raise RuntimeError
                integrator = MTSVVVRIntegrator(temperature*kelvin, collision/picosecond,
                                               timestep*femtosecond, self.system, ninnersteps=int(timestep/faststep))
            else:
                if rpmd_beads > 0:
                    logger.info("Creating RPMD integrator with %i beads.\n" % rpmd_beads)
                    self.tdiv = rpmd_beads
                    integrator = RPMDIntegrator(rpmd_beads, temperature*kelvin, collision/picosecond, timestep*femtosecond)
                else:
                    integrator = LangevinIntegrator(temperature*kelvin, collision/picosecond, timestep*femtosecond)
        else:
            ## If no temperature control, default to the Verlet integrator.
            if rpmd_beads > 0:
                logger.error("No RPMD integrator without temperature control.\n")
                raise RuntimeError
            if mts: warn_once("No multiple timestep integrator without temperature control.")
            integrator = VerletIntegrator(timestep*femtoseconds)

        ## Add the barostat.
        if pressure != None:
            if anisotropic:
                barostat = MonteCarloAnisotropicBarostat([pressure, pressure, pressure]*atmospheres,
                                                         temperature*kelvin, nbarostat)
            else:
                barostat = MonteCarloBarostat(pressure*atmospheres, temperature*kelvin, nbarostat)
        if self.pbc and pressure != None: self.system.addForce(barostat)
        elif pressure != None: warn_once("Pressure is ignored because pbc is set to False.")

        ## Set up for energy component analysis.
        GrpTogether = ['AmoebaGeneralizedKirkwoodForce', 'AmoebaMultipoleForce', 'AmoebaWcaDispersionForce']
        GrpNums = {}
        if not mts:
            for j in self.system.getForces():
                i = -1
                if j.__class__.__name__ in GrpTogether:
                    for k in GrpNums:
                        if k in GrpTogether:
                            i = GrpNums[k]
                            break
                if i == -1: i = len(set(GrpNums.values()))
                GrpNums[j.__class__.__name__] = i
                j.setForceGroup(i)

        ## If virtual particles are used with AMOEBA...
        SetAmoebaVirtualExclusions(self.system)
        
        ## Finally create the simulation object.
        self.simulation = Simulation(self.mod.topology, self.system, integrator, self.platform)
        
        ## Print platform properties.
        # logger.info("I'm using the platform %s\n" % self.simulation.context.getPlatform().getName())
        # printcool_dictionary({i:self.simulation.context.getPlatform().getPropertyValue(self.simulation.context,i) \
        #                           for i in self.simulation.context.getPlatform().getPropertyNames()}, \
        #                          title="Platform %s has properties:" % self.simulation.context.getPlatform().getName())

    def update_simulation(self, **kwargs):

        """ 
        Create the simulation object, or update the force field
        parameters in the existing simulation object.  This should be
        run when we write a new force field XML file.
        """
        if len(kwargs) > 0:
            self.simkwargs = kwargs
        if type(self.ffxml) == list:
            self.forcefield = ForceField(*self.ffxml)
        else:
            self.forcefield = ForceField(self.ffxml)
        self.mod = Modeller(self.pdb.topology, self.pdb.positions)
        self.mod.addExtraParticles(self.forcefield)
        # printcool_dictionary(self.mmopts, title="Creating/updating simulation in engine %s with system settings:" % (self.name))
        self.system = self.forcefield.createSystem(self.mod.topology, **self.mmopts)
        self.vsinfo = PrepareVirtualSites(self.system)
        self.nbcharges = np.zeros(self.system.getNumParticles())
        for i in self.system.getForces():
            if isinstance(i, NonbondedForce):
                self.nbcharges = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])

        #----
        # If the virtual site parameters have changed,
        # the simulation object must be remade.
        #----
        vsprm = GetVirtualSiteParameters(self.system)
        if hasattr(self,'vsprm') and len(self.vsprm) > 0 and np.max(np.abs(vsprm - self.vsprm)) != 0.0:
            if hasattr(self, 'simulation'): 
                delattr(self, 'simulation')
        self.vsprm = vsprm.copy()

        if hasattr(self, 'simulation'):
            UpdateSimulationParameters(self.system, self.simulation)
        else:
            self.create_simulation(**self.simkwargs)

    def set_positions(self, shot=0, traj=None):
        
        """
        Set the positions and periodic box vectors to one of the
        stored coordinates.  

        *** NOTE: If you run a MD simulation, then the coordinates are
        overwritten by the MD trajectory. ***
        """
        #----
        # Ideally the virtual site parameters would be copied but they're not.
        # Instead we update the vsite positions manually.
        #----
        # if self.FF.rigid_water:
        #     simulation.context.applyConstraints(1e-8)
        # else:
        #     simulation.context.computeVirtualSites()
        #----
        # NOTE: Periodic box vectors must be set FIRST
        if self.pbc:
            self.simulation.context.setPeriodicBoxVectors(*self.xyz_omms[shot][1])
        # self.simulation.context.setPositions(ResetVirtualSites(self.xyz_omms[shot][0], self.system))
        # self.simulation.context.setPositions(ResetVirtualSites_fast(self.xyz_omms[shot][0], self.vsinfo))
        self.simulation.context.setPositions(self.xyz_omms[shot][0])
        self.simulation.context.computeVirtualSites()

    def compute_volume(self, box_vectors):
        """ Compute the total volume of an OpenMM system. """
        [a,b,c] = box_vectors
        A = np.array([a/a.unit, b/a.unit, c/a.unit])
        # Compute volume of parallelepiped.
        volume = np.linalg.det(A) * a.unit**3
        return volume
    
    def compute_mass(self, system):
        """ Compute the total mass of an OpenMM system. """
        mass = 0.0 * amu
        for i in range(system.getNumParticles()):
            mass += system.getParticleMass(i)
        return mass

    def evaluate_one_(self, force=False, dipole=False):
        # Perform a single point calculation on the current geometry.        
        State = self.simulation.context.getState(getPositions=dipole, getEnergy=True, getForces=force)
        Result = {}
        Result["Energy"] = State.getPotentialEnergy() / kilojoules_per_mole
        if force: 
            Force = list(np.array(State.getForces() / kilojoules_per_mole * nanometer).flatten())
            # Extract forces belonging to real atoms only
            Result["Force"] = np.array(list(itertools.chain(*[Force[3*i:3*i+3] for i in range(len(Force)/3) if self.AtomMask[i]])))
        if dipole: Result["Dipole"] = get_dipole(self.simulation, q=self.nbcharges, mass=self.AtomLists['Mass'], positions=State.getPositions())
        return Result

    def evaluate_(self, force=False, dipole=False, traj=False):

        """ 
        Utility function for computing energy, and (optionally) forces and dipoles using OpenMM. 
        
        Inputs:
        force: Switch for calculating the force.
        dipole: Switch for calculating the dipole.
        traj: Trajectory (listing of coordinate and box 2-tuples).  If provide, will loop over these snapshots.
        Otherwise will do a single point evaluation at the current geometry.

        Outputs:
        Result: Dictionary containing energies, forces and/or dipoles.
        """

        self.update_simulation()
        # If trajectory flag set to False, perform a single-point calculation.
        if not traj: return evaluate_one_(force, dipole)
        Energies = []
        Forces = []
        Dipoles = []
        for I in range(len(self.xyz_omms)):
            self.set_positions(I)
            R1 = self.evaluate_one_(force, dipole)
            Energies.append(R1["Energy"])
            if force: Forces.append(R1["Force"])
            if dipole: Dipoles.append(R1["Dipole"])
        # Compile it all into the dictionary object
        Result = OrderedDict()
        Result["Energy"] = np.array(Energies)
        if force: Result["Force"] = np.array(Forces)
        if dipole: Result["Dipole"] = np.array(Dipoles)
        return Result

    def energy_one(self, shot):
        self.set_positions(shot)
        return self.evaluate_()["Energy"]

    def energy_force_one(self):
        self.set_positions(shot)
        Result = self.evaluate_(force=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy(self):
        return self.evaluate_(traj=True)["Energy"]

    def energy_force(self):
        """ Loop through the snapshots and compute the energies and forces using OpenMM. """
        Result = self.evaluate_(force=True, traj=True)
        E = Result["Energy"].reshape(-1,1)
        F = Result["Force"]
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Force"]))

    def energy_dipole(self):
        """ Loop through the snapshots and compute the energies and forces using OpenMM. """
        Result = self.evaluate_(dipole=True, traj=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Dipole"]))

    def normal_modes(self, shot=0, optimize=True):
        logger.error("OpenMM cannot do normal mode analysis\n")
        raise NotImplementedError

    def optimize(self, shot=0, crit=1e-4):

        """ Optimize the geometry and align the optimized geometry to the starting geometry, and return the RMSD. """
        
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
        mod.addExtraParticles(self.forcefield)
        # self.simulation.context.setPositions(ResetVirtualSites(mod.getPositions(), self.system))
        self.simulation.context.setPositions(ResetVirtualSites_fast(mod.getPositions(), self.vsinfo))
        return E, M.ref_rmsd(0)[1]

    def multipole_moments(self, shot=0, optimize=True, polarizability=False):

        """ Return the multipole moments of the i-th snapshot in Debye and Buckingham units. """

        self.update_simulation()

        if polarizability:
            logger.error("Polarizability calculation is available in TINKER only.\n")
            raise NotImplementedError

        if optimize: self.optimize(shot)
        else: self.set_positions(shot)

        moments = get_multipoles(self.simulation)
        
        dipole_dict = OrderedDict(zip(['x','y','z'], moments[:3]))
        quadrupole_dict = OrderedDict(zip(['xx','xy','yy','xz','yz','zz'], moments[3:10]))

        calc_moments = OrderedDict([('dipole', dipole_dict), ('quadrupole', quadrupole_dict)])

        return calc_moments

    def energy_rmsd(self, shot=0, optimize=True):

        """ Calculate energy of the 1st structure (optionally minimize and return the minimized energy and RMSD). In kcal/mol. """

        self.update_simulation()

        if self.platname in ['CUDA', 'OpenCL'] and self.precision in ['single', 'mixed']:
            crit = 1e-4
        else:
            crit = 1e-6

        rmsd = 0.0
        if optimize: 
            E, rmsd = self.optimize(shot, crit=crit)
        else:
            self.set_positions(shot)
            E = self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalories_per_mole)

        return E, rmsd

    def interaction_energy(self, fraga, fragb):
        
        """ Calculate the interaction energy for two fragments. """

        self.update_simulation()

        if self.name == 'A' or self.name == 'B':
            logger.error("Don't name the engine A or B!\n")
            raise RuntimeError

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

    def molecular_dynamics(self, nsteps, timestep, temperature=None, pressure=None, nequil=0, nsave=1000, minimize=True, anisotropic=False, save_traj=False, verbose=False, **kwargs):
        
        """
        Method for running a molecular dynamics simulation.  

        Required arguments:
        nsteps      = (int)   Number of total time steps
        timestep    = (float) Time step in FEMTOSECONDS
        temperature = (float) Temperature control (Kelvin)
        pressure    = (float) Pressure control (atmospheres)
        nequil      = (int)   Number of additional time steps at the beginning for equilibration
        nsave       = (int)   Step interval for saving and printing data
        minimize    = (bool)  Perform an energy minimization prior to dynamics

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        """

        if float(int(float(nequil)/float(nsave))) != float(nequil)/float(nsave):
            logger.error("Please set nequil to an integer multiple of nsave\n")
            raise RuntimeError
        iequil = nequil/nsave

        if float(int(float(nsteps)/float(nsave))) != float(nsteps)/float(nsave):
            logger.error("Please set nsteps to an integer multiple of nsave\n")
            raise RuntimeError
        isteps = nsteps/nsave

        if hasattr(self, 'simulation'):
            logger.warning('Deleting the simulation object and re-creating for MD\n')
            delattr(self, 'simulation')

        self.update_simulation(timestep=timestep, temperature=temperature, pressure=pressure, anisotropic=anisotropic, **kwargs)
        self.set_positions()

        # Minimize the energy.
        if minimize:
            if verbose: logger.info("Minimizing the energy... (starting energy % .3f kJ/mol)" % 
                                    self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole))
            self.simulation.minimizeEnergy()
            if verbose: logger.info("Done (final energy % .3f kJ/mol)\n" % 
                                    self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole))

        # Serialize the system if we want.
        Serialize = 0
        if Serialize:
            serial = XmlSerializer.serializeSystem(system)
            with wopen('%s_system.xml' % phase) as f: f.write(serial)

        # Determine number of degrees of freedom; the center of mass motion remover is also a constraint.
        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        
        # Compute total mass.
        self.mass = self.compute_mass(self.system).in_units_of(gram / mole) /  AVOGADRO_CONSTANT_NA

        # Determine number of degrees of freedom.
        self.ndof = 3*(self.system.getNumParticles() - sum([self.system.isVirtualSite(i) for i in range(self.system.getNumParticles())])) \
            - self.system.getNumConstraints() - 3*self.pbc

        # Initialize statistics.
        edecomp = OrderedDict()
        # Stored coordinates, box vectors
        self.xyz_omms = []
        # Densities, potential and kinetic energies, box volumes, dipole moments
        Rhos = []
        Potentials = []
        Kinetics = []
        Volumes = []
        Dips = []
        Temps = []
        #========================#
        # Now run the simulation #
        #========================#
        # Initialize velocities.
        self.simulation.context.setVelocitiesToTemperature(temperature*kelvin)
        # Equilibrate.
        if iequil > 0: 
            if verbose: logger.info("Equilibrating...\n")
            if self.pbc:
                if verbose: logger.info("%6s %9s %9s %13s %10s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)", "Vol(nm^3)", "Rho(kg/m^3)"))
            else:
                if verbose: logger.info("%6s %9s %9s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)"))
        for iteration in range(-1, iequil):
            if iteration >= 0:
                self.simulation.step(nsave)
            state = self.simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=False,getForces=False)
            kinetic = state.getKineticEnergy()/self.tdiv
            potential = state.getPotentialEnergy()
            if self.pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (self.mass / volume).in_units_of(kilogram / meter**3)
            else:
                volume = 0.0 * nanometers ** 3
                density = 0.0 * kilogram / meter ** 3
            kinetic_temperature = 2.0 * kinetic / kB / self.ndof # (1/2) ndof * kB * T = KE
            if self.pbc:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f %10.4f %13.4f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                                     kinetic_temperature / kelvin, potential / kilojoules_per_mole,
                                                                                     volume / nanometers**3, density / (kilogram / meter**3)))
            else:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                       kinetic_temperature / kelvin, potential / kilojoules_per_mole))
        # Collect production data.
        if verbose: logger.info("Production...\n")
        if self.pbc:
            if verbose: logger.info("%6s %9s %9s %13s %10s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)", "Vol(nm^3)", "Rho(kg/m^3)"))
        else:
            if verbose: logger.info("%6s %9s %9s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)"))
        if save_traj:
            self.simulation.reporters.append(PDBReporter('%s-md.pdb' % self.name, nsteps))
            self.simulation.reporters.append(DCDReporter('%s-md.dcd' % self.name, nsave))
        for iteration in range(-1, isteps):
            # Propagate dynamics.
            if iteration >= 0: self.simulation.step(nsave)
            # Compute properties.
            state = self.simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=False,getForces=False)
            kinetic = state.getKineticEnergy()/self.tdiv
            potential = state.getPotentialEnergy()
            kinetic_temperature = 2.0 * kinetic / kB / self.ndof
            if self.pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (self.mass / volume).in_units_of(kilogram / meter**3)
            else:
                box_vectors = None
                volume = 0.0 * nanometers ** 3
                density = 0.0 * kilogram / meter ** 3
            self.xyz_omms.append([state.getPositions(), box_vectors])
            # Perform energy decomposition.
            for comp, val in energy_components(self.simulation).items():
                if comp in edecomp:
                    edecomp[comp].append(val)
                else:
                    edecomp[comp] = [val]
            if self.pbc:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f %10.4f %13.4f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                                     kinetic_temperature / kelvin, potential / kilojoules_per_mole,
                                                                                     volume / nanometers**3, density / (kilogram / meter**3)))
            else:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                       kinetic_temperature / kelvin, potential / kilojoules_per_mole))
            Temps.append(kinetic_temperature / kelvin)
            Rhos.append(density.value_in_unit(kilogram / meter**3))
            Potentials.append(potential / kilojoules_per_mole)
            Kinetics.append(kinetic / kilojoules_per_mole)
            Volumes.append(volume / nanometer**3)
            Dips.append(get_dipole(self.simulation,positions=self.xyz_omms[-1][0]))
        Rhos = np.array(Rhos)
        Potentials = np.array(Potentials)
        Kinetics = np.array(Kinetics)
        Volumes = np.array(Volumes)
        Dips = np.array(Dips)
        Ecomps = OrderedDict([(key, np.array(val)) for key, val in edecomp.items()])
        Ecomps["Potential Energy"] = np.array(Potentials)
        Ecomps["Kinetic Energy"] = np.array(Kinetics)
        Ecomps["Temperature"] = np.array(Temps)
        Ecomps["Total Energy"] = np.array(Potentials) + np.array(Kinetics)
        # Initialized property dictionary.
        prop_return = OrderedDict()
        prop_return.update({'Rhos': Rhos, 'Potentials': Potentials, 'Kinetics': Kinetics, 'Volumes': Volumes, 'Dips': Dips, 'Ecomps': Ecomps})
        return prop_return

class Liquid_OpenMM(Liquid):
    """ Condensed phase property matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'force_cuda',forceprint=True)
        # Enable multiple timestep integrator
        self.set_option(tgt_opts,'mts_integrator',forceprint=True)
        # Enable ring polymer MD
        self.set_option(options,'rpmd_beads',forceprint=True)
        # OpenMM precision
        self.set_option(tgt_opts,'openmm_precision','precision',default="mixed")
        # OpenMM platform
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA")
        # Name of the liquid coordinate file.
        self.set_option(tgt_opts,'liquid_coords',default='liquid.pdb',forceprint=True)
        # Name of the gas coordinate file.
        self.set_option(tgt_opts,'gas_coords',default='gas.pdb',forceprint=True)
        # Set the number of steps between MC barostat adjustments.
        self.set_option(tgt_opts,'mc_nbarostat')
        # Class for creating engine object.
        self.engine_ = OpenMM
        # Name of the engine to pass to npt.py.
        self.engname = "openmm"
        # Command prefix.
        self.nptpfx = "bash runcuda.sh"
        # Extra files to be linked into the temp-directory.
        self.nptfiles = []
        # Set some options for the polarization correction calculation.
        self.gas_engine_args = {}
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ['runcuda.sh']
        # Initialize the base class.
        super(Liquid_OpenMM,self).__init__(options,tgt_opts,forcefield)
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output = ['liquid-md.pdb', 'liquid-md.dcd']

class AbInitio_OpenMM(AbInitio):
    """ Force and energy matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'pdb',default="conf.pdb")
        self.set_option(tgt_opts,'coords',default="all.gro")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
        self.engine_ = OpenMM
        ## Initialize base class.
        super(AbInitio_OpenMM,self).__init__(options,tgt_opts,forcefield)

class BindingEnergy_OpenMM(BindingEnergy):
    """ Binding energy matching using OpenMM. """

    def __init__(self,options,tgt_opts,forcefield):
        self.engine_ = OpenMM
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
        ## Initialize base class.
        super(BindingEnergy_OpenMM,self).__init__(options,tgt_opts,forcefield)

class Interaction_OpenMM(Interaction):
    """ Interaction matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="all.pdb")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
        self.engine_ = OpenMM
        ## Initialize base class.
        super(Interaction_OpenMM,self).__init__(options,tgt_opts,forcefield)

class Moments_OpenMM(Moments):
    """ Multipole moment matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        self.set_option(tgt_opts,'coords',default="input.pdb")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
        self.engine_ = OpenMM
        ## Initialize base class.
        super(Moments_OpenMM,self).__init__(options,tgt_opts,forcefield)
