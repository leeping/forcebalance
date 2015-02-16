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
logger = getLogger(__name__)
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import simtk.openmm._openmm as _openmm
except:
    pass

def get_mask(grps):
    """ Given a list of booleans [1, 0, 1] return the bitmask that sets 
    these force groups appropriately in Context.getState(). Any values
    not provided are defaulted to 1.  """
    mask = 0
    for i, j in enumerate(grps):
        # print i, j, 2**i
        mask += 2**i*j
    for k in range(i+1, 31):
        # print k, 1, 2**k
        mask += 2**k
    return mask

def energy_components(Sim, verbose=False):
    # Before using energy_components(), make sure each Force is set to a different group.
    EnergyTerms = OrderedDict()
    if type(Sim.integrator) in [LangevinIntegrator, VerletIntegrator]:
        for i in range(Sim.system.getNumForces()):
            EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
    return EnergyTerms

def evaluate_potential(Sim):
    # Returns potential averaged over copies for PIMD, potential energy
    # otherwise. 
    if isinstance(Sim.integrator, RPMDIntegrator):
        PE = 0.0 * kilojoule/mole
        P = Sim.integrator.getNumCopies()
        for i in range(P):
            PE += Sim.integrator.getState(i,getEnergy=True).getPotentialEnergy() / P
        return PE
    else:
        return Sim.context.getState(getEnergy=True).getPotentialEnergy() 

def evaluate_kinetic(Sim, props):
    # A primitive quantum K.E. estimator for PIMD simulation. 
    # Returns classical K.E. in a classical simulation. 
    if isinstance(Sim.integrator, RPMDIntegrator):
        spring_term = 0.0
        hbar = 0.06350780 * nanometer**2*dalton/picosecond           # 1 nm**2 * Da / ps**2 is 1 kJ/mol    
        kb = 0.00831446 * nanometer**2*dalton/(picosecond**2*kelvin)
        T = props['T']                                               # Note: this method call does NOT return instantaneous temp, but rather whatever the integrator was set to
        P = props['P']
        const1 = (kb*T)**2*P/(2.0*hbar**2)
        const2 = (kb*T)**3*P/hbar**2
        for i in range(P):
            j = (i+1) % P
            diff = np.array(props['Positions'][j])-np.array(props['Positions'][i])
            spring_term -= np.dot((diff*diff).sum(axis=1), props['Masses'])
        spring_term = Quantity(spring_term, nanometer**2*dalton)
        return spring_term*const1, spring_term*const2 
    else:
        return Sim.context.getState(getEnergy=True).getKineticEnergy()

def centroid_kinetic(Sim, props):
    # Centroid quantum K.E. estimator for RPMD simulation. 
    # Returns classical K.E. in classical simulation.
    if isinstance(Sim.integrator, RPMDIntegrator):
        CV_second_term = 0.0
        P = props['P']
        N = props['N']
        centroid = np.array([[0.0,0.0,0.0]]*N)
        for i in range(P):
            centroid += np.array(props['Positions'][i]) / P
        for i in range(P):
            diff = np.array(props['Positions'][i]) - centroid
            diff[np.where(props['Vsites'])[0],:] = 0.0
            derivative = -np.array(props['States'][i].getForces().value_in_unit(kilojoules_per_mole/nanometer))
            CV_second_term += np.sum(np.multiply(diff,derivative))
        return Quantity(CV_second_term*0.5/P, kilojoules_per_mole)
    else:
        Sim.context.getState(getEnergy=True).getKineticEnergy()

def get_forces(Sim):
    """Return forces on each atom or forces averaged over all copies in case of RPMD."""
    if isinstance(Sim.integrator, RPMDIntegrator):
        P = Sim.integrator.getNumCopies()
        frcs = Sim.integrator.getState(0,getForces=True).getForces() / P
        for i in range(1,P):
            frcs += Sim.integrator.getState(i,getForces=True).getForces() / P
        return frcs
    else:
        State = Sim.context.getState(getPositions=True, getEnergy=True, getForces=True)
        return State.getForces() 

def rpmd_dips(Sim, qvals, masses):
    """Return dipole moments averaged over all copies of the system."""
    rpmdIntegrator = Sim.context.getIntegrator()
    temp_dips = []
    for i in range(rpmdIntegrator.getNumCopies()):
        temp_positions = rpmdIntegrator.getState(i,getPositions=True).getPositions()
        temp_dips.append(get_dipole(Sim, q=qvals, mass=masses, positions=temp_positions))
    dip_avg = [sum(col) / float(len(col)) for col in zip(*temp_dips)]
    return dip_avg
 
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

def GetSystemConstraints(system):
    """Return boolean value indicating whether system has constraints. RPMD simulations should have no constraints."""
    return system.getNumConstraints() > 0

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

def CopyGBSAOBCParameters(src, dest):
    dest.setSolventDielectric(src.getSolventDielectric())
    dest.setSoluteDielectric(src.getSoluteDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))

def CopyCustomNonbondedParameters(src, dest):
    '''
    copy whatever updateParametersInContext can update:
        per-particle parameters
    '''
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i, list(src.getParticleParameters(i)))

def CopyCustomBondedParameters(src, dest):
    '''
    Copy whatever update parameters in context can 
    '''
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i, *src.getBondParameters(i))

def do_nothing(src, dest):
    return

def CopySystemParameters(src,dest):
    """Copy parameters from one system (i.e. that which is created by a new force field)
    to another system (i.e. the one stored inside the Target object).
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
               'CustomNonbondedForce':CopyCustomNonbondedParameters,
               'GBSAOBCForce':CopyGBSAOBCParameters,
               'CMMotionRemover':do_nothing,
               'CustomBondForce':CopyCustomBondedParameters}
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
        if isinstance(dest_simulation.system.getForce(i), CustomNonbondedForce):
            force = src_system.getForce(i)
            for j in range(force.getNumGlobalParameters()):
                pName = force.getGlobalParameterName(j)
                pValue = force.getGlobalParameterDefaultValue(j)
                dest_simulation.context.setParameter(pName, pValue)


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

def AddVirtualSiteBonds(mod, ff):
    # print "In AddVirtualSiteBonds"
    for ir, R in enumerate(list(mod.topology.residues())):
        A = list(R.atoms())
        # print "Residue", ir, ":", R.name
        for vs in ff._templates[R.name].virtualSites:
            vi = vs.index
            for ai in vs.atoms:
                bi = sorted([A[ai], A[vi]])
                # print "Adding Bond", ai, vi
                mod.topology.addBond(*bi)

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
        self.valkwd = ['ffxml', 'pdb', 'platname', 'precision', 'mmopts', 'vsite_bonds', 'implicit_solvent']
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
        if 'implicit_solvent' in kwargs:
            # Force implicit solvent to either On or Off.
            self.ism = kwargs['implicit_solvent']
        else:
            self.ism = None

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
            if pdbfnm == None and kwargs['coords'].endswith('.pdb'):
                pdbfnm = kwargs['coords']
        else:
            logger.error('Must provide either a molecule object or coordinate file.\n')
            raise RuntimeError
        
        # If the PDB file exists, then it is copied directly to create
        # the OpenMM pdb object rather than being written by the
        # Molecule class.
        if pdbfnm != None:
            self.abspdb = os.path.abspath(pdbfnm)
            mpdb = Molecule(pdbfnm)
            for i in ["chain", "atomname", "resid", "resname", "elem"]:
                self.mol.Data[i] = mpdb.Data[i]

    def prepare(self, pbc=False, mmopts={}, **kwargs):

        """ 
        Prepare the calculation.  Note that we don't create the
        Simulation object yet, because that may depend on MD
        integrator parameters, thermostat, barostat etc.
        """
        # Introduced to attempt to fix a bug, but didn't work,
        # Might be sensible code anyway.
        # if hasattr(self.mol, 'boxes') and not pbc:
        #     del self.mol.Data['boxes']
        # if pbc and not hasattr(self.mol, 'boxes'):
        #     logger.error('Requested periodic boundary conditions but coordinate file contains no boxes')
        #     raise RuntimeError
        ## Create the OpenMM PDB object.
        if hasattr(self, 'abspdb'):
            self.pdb = PDBFile(self.abspdb)
        else:
            pdb1 = "%s-1.pdb" % os.path.splitext(os.path.basename(self.mol.fnm))[0]
            self.mol[0].write(pdb1)
            self.pdb = PDBFile(pdb1)
            os.unlink(pdb1)
        
        ## Create the OpenMM ForceField object.
        if hasattr(self, 'FF'):
            self.ffxml = [self.FF.openmmxml]
            self.forcefield = ForceField(os.path.join(self.root, self.FF.ffdir, self.FF.openmmxml))
        else:
            self.ffxml = listfiles(kwargs.get('ffxml'), 'xml', err=True)
            self.forcefield = ForceField(*self.ffxml)

        ## Create bonds between virtual sites and their host atoms.
        ## This is mainly for setting up AMOEBA multipoles.
        self.vbonds = kwargs.get('vsite_bonds', 0)

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
            self.mmopts.setdefault('nonbondedMethod', CutoffPeriodic)
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

    def create_simulation(self, timestep=1.0, faststep=0.25, temperature=None, pressure=None, anisotropic=False, mts=False, collision=1.0, nbarostat=25, rpmd_opts=[], **kwargs):

        """
        Create simulation object.  Note that this also takes in some
        options pertinent to system setup, including the type of MD
        integrator and type of pressure control.
        """
        # Divisor for the temperature (RPMD sets it to nonzero.)
        self.tdiv = 1
        
        # Boolean flag indicating an RPMD simulation
        self.rpmd = len(rpmd_opts)>0
        rpmd_opts = [int(i) for i in rpmd_opts]

        ## Determine the integrator.
        if temperature:
            ## If temperature control is turned on, then run Langevin dynamics.
            if mts:
                if len(rpmd_opts) > 0:
                    raise RuntimeError("No multiple timestep integrator without temperature control.")
                integrator = MTSVVVRIntegrator(temperature*kelvin, collision/picosecond,
                                               timestep*femtosecond, self.system, ninnersteps=int(timestep/faststep))
            else:
                if len(rpmd_opts) > 0:
                    self.tdiv = int(rpmd_opts[0])
                    if len(rpmd_opts) == 1:
                        logger.info("Creating RPMD integrator with %i beads.\n" % int(rpmd_opts[0]))
                        integrator = RPMDIntegrator(int(rpmd_opts[0]), temperature*kelvin, collision/picosecond, timestep*femtosecond)
                    elif len(rpmd_opts) == 2:
                        contract = False
                        for frc in self.system.getForces():
                            if any([isinstance(frc, fc) for fc in [NonbondedForce, AmoebaMultipoleForce, AmoebaVdwForce, CustomNonbondedForce]]):
                                contract = True
                                frc.setForceGroup(1)
                        if contract:
                            logger.info("Creating RPMD integrator with %i beads (NB forces contracted to %i).\n" % (int(rpmd_opts[0]), int(rpmd_opts[1])))
                            integrator = RPMDIntegrator(int(rpmd_opts[0]), temperature*kelvin, collision/picosecond, timestep*femtosecond, {1:int(rpmd_opts[1])})
                        else:
                            logger.info("Creating RPMD integrator with %i beads (no NB forces to contract).\n" % (int(rpmd_opts[0])))
                            integrator = RPMDIntegrator(int(rpmd_opts[0]), temperature*kelvin, collision/picosecond, timestep*femtosecond)
                    elif len(rpmd_opts) == 3:
                        contract = False
                        contract_recip = False
                        for frc in self.system.getForces():
                            if any([isinstance(frc, fc) for fc in [NonbondedForce, AmoebaMultipoleForce, AmoebaVdwForce, CustomNonbondedForce]]):
                                contract = True
                                frc.setForceGroup(1)
                                if isinstance(frc, NonbondedForce):
                                    contract_recip = True
                                    frc.setReciprocalSpaceForceGroup(2)
                        if contract_recip:
                            logger.info("Creating RPMD integrator with %i beads (NB/Recip forces contracted to %i/%i).\n" % (int(rpmd_opts[0]), int(rpmd_opts[1]), int(rpmd_opts[2])))
                            integrator = RPMDIntegrator(int(rpmd_opts[0]), temperature*kelvin, collision/picosecond, timestep*femtosecond, {1:int(rpmd_opts[1]), 2:int(rpmd_opts[2])})
                        elif contract:
                            logger.info("Creating RPMD integrator with %i beads (NB forces contracted to %i, no Recip).\n" % (int(rpmd_opts[0]), int(rpmd_opts[1])))
                            integrator = RPMDIntegrator(int(rpmd_opts[0]), temperature*kelvin, collision/picosecond, timestep*femtosecond, {1:int(rpmd_opts[1])})
                        else:
                            logger.info("Creating RPMD integrator with %i beads (no NB forces to contract).\n" % (int(rpmd_opts[0])))
                            integrator = RPMDIntegrator(int(rpmd_opts[0]), temperature*kelvin, collision/picosecond, timestep*femtosecond)
                    else:
                        raise RuntimeError("Please provide a list of length 1, 2, or 3 to rpmd_opts")
                else:
                    integrator = LangevinIntegrator(temperature*kelvin, collision/picosecond, timestep*femtosecond)
        else:
            ## If no temperature control, default to the Verlet integrator.
            if len(rpmd_opts) > 0:
                raise RuntimeError("No RPMD integrator without temperature control.")
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
        GrpTogether = ['AmoebaGeneralizedKirkwoodForce', 'AmoebaMultipoleForce','AmoebaWcaDispersionForce',
                        'CustomNonbondedForce',  'NonbondedForce']
        GrpNums = {}
        if not mts and len(rpmd_opts) == 0:
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
        #fffff.write(XmlSerializer.serialize(self.simulation.system))
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
        self.forcefield = ForceField(*self.ffxml)
        # OpenMM classes for force generators
        ismgens = [forcefield.AmoebaGeneralizedKirkwoodGenerator, forcefield.AmoebaWcaDispersionGenerator,
                     forcefield.CustomGBGenerator, forcefield.GBSAOBCGenerator, forcefield.GBVIGenerator]
        if self.ism != None:
            if self.ism == False:
                self.forcefield._forces = [f for f in self.forcefield._forces if not any([isinstance(f, f_) for f_ in ismgens])]
            elif self.ism == True:
                if len([f for f in self.forcefield._forces if any([isinstance(f, f_) for f_ in ismgens])]) == 0:
                    logger.error("There is no implicit solvent model!\n")
                    raise RuntimeError
        self.mod = Modeller(self.pdb.topology, self.pdb.positions)
        self.mod.addExtraParticles(self.forcefield)
        # Add bonds for virtual sites. (Experimental)
        if self.vbonds: AddVirtualSiteBonds(self.mod, self.forcefield)
        # printcool_dictionary(self.mmopts, title="Creating/updating simulation in engine %s with system settings:" % (self.name))
        # for b in list(self.mod.topology.bonds()):
        #     print b[0].index, b[1].index
        if 'rpmd_opts' in kwargs:
            self.mmopts['rigidWater'] = False
            self.mmopts['constraints'] = 'None'
        self.system = self.forcefield.createSystem(self.mod.topology, **self.mmopts)
        self.vsinfo = PrepareVirtualSites(self.system)
        self.nbcharges = np.zeros(self.system.getNumParticles())
        for i in self.system.getForces():
            if isinstance(i, NonbondedForce):
                self.nbcharges = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])
                if not any([isinstance(fc, CustomNonbondedForce) for fc in self.system.getForces()]):
                    i.setNonbondedMethod(4)
        if any([isinstance(fc, NonbondedForce) for fc in self.system.getForces()]) and any([isinstance(fc, CustomNonbondedForce) for fc in self.system.getForces()]):
        # Case of fitting the softer potential
            for i in self.system.getForces():
                if isinstance(i, NonbondedForce):
                    i.setNonbondedMethod(4)
                    i.setUseSwitchingFunction(False)
                    #logger.info('NonbondedForce\n')
                    #logger.info('Nonbonded method\n')
                    #logger.info(i.getNonbondedMethod())
                    #logger.info('Cutoff distance\n')
                    #logger.info(i.getCutoffDistance())
                    #logger.info('Use switching function\n')
                    #logger.info(i.getUseSwitchingFunction())
                    #logger.info('Switching distance\n')
                    #logger.info(i.getSwitchingDistance())
                    #logger.info('Dispersion Correction\n')
                    #logger.info(i.getUseDispersionCorrection())
                elif isinstance(i, CustomNonbondedForce):
                    #logger.info('')
                    i.setNonbondedMethod(4)
                    i.setUseLongRangeCorrection(True)
                    #logger.info('CustomNonbondedForce\n')
                    #logger.info('Nonbonded method\n')
                    #logger.info(i.getNonbondedMethod())
                    #logger.info('Cutoff distance\n')
                    #logger.info(i.getCutoffDistance())
                    #logger.info('Use switching function\n')
                    #logger.info(i.getUseSwitchingFunction())
                    #logger.info('Switching distance\n')
                    #logger.info(i.getSwitchingDistance())
                    #logger.info('Correction\n')
                    #logger.info(i.getUseLongRangeCorrection())
        #----
        # If the virtual site parameters have changed,
        # the simulation object must be remade.
        #----
        vsprm = GetVirtualSiteParameters(self.system)
        if hasattr(self,'vsprm') and len(self.vsprm) > 0 and np.max(np.abs(vsprm - self.vsprm)) != 0.0:
            if hasattr(self, 'simulation'): 
                delattr(self, 'simulation')
        self.vsprm = vsprm.copy()
        #----
	# If number of of constraints in the new
	# system differs from simulation's system,
	# similarly remake the simulation object
	#----
        if hasattr(self, 'simulation'):
            new_system_constraints = GetSystemConstraints(self.system)
            simulation_system_constraints = GetSystemConstraints(self.simulation.system)
            if new_system_constraints != simulation_system_constraints: delattr(self, 'simulation')
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
            if not hasattr(self, 'xyz_rpmd'):
                self.simulation.context.setPeriodicBoxVectors(*self.xyz_omms[shot][1])
            else:
                self.simulation.context.setPeriodicBoxVectors(*self.xyz_rpmd[shot][1])
        # self.simulation.context.setPositions(ResetVirtualSites(self.xyz_omms[shot][0], self.system))
        # self.simulation.context.setPositions(ResetVirtualSites_fast(self.xyz_omms[shot][0], self.vsinfo))
        if not hasattr(self, 'xyz_rpmd'):
            self.simulation.context.setPositions(self.xyz_omms[shot][0])
            self.simulation.context.computeVirtualSites()
            #-----------------
            # For RPMD, initially it is fine to set all beads using coordinates in xyz_omms. We also need to 
            # compute virtual sites by processing each copy via the context. We should NOT redo this each 
            # time set_positions() is called, as by then the virtual sites are already contained in xyz_rpmd.
            #-----------------
            if isinstance(self.simulation.integrator, RPMDIntegrator):
                rpmdIntegrator = self.simulation.context.getIntegrator()
                for i in range(rpmdIntegrator.getNumCopies()):
                    temp_positions = self.xyz_omms[shot][0]
                    self.simulation.context.setPositions(temp_positions)
                    self.simulation.context.computeVirtualSites()
                    posWithVsites = self.simulation.context.getState(getPositions=True).getPositions()
                    rpmdIntegrator.setPositions(i,posWithVsites)
        else:
            rpmdIntegrator = self.simulation.context.getIntegrator()
            for i in range(rpmdIntegrator.getNumCopies()):
                temp_positions = self.xyz_rpmd[shot][0][i]
                rpmdIntegrator.setPositions(i,temp_positions)
    
    def get_charges(self):
        logger.error('OpenMM engine does not have get_charges (should be trivial to implement however.)')
        raise NotImplementedError

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

    def rpmd_cv(self, traj=True):
        self.update_simulation()
        if not hasattr(self, 'rpmd_frame_props'):
            rpmd_frame_props = {}
            rpmd_frame_props['States'] = []
            vsites = []
            for i in range(self.simulation.system.getNumParticles()):
                if self.simulation.system.isVirtualSite(i):
                    vsites.append(True)
                else:
                    vsites.append(False)
            mass_matrix = np.array([])
            for i in range(self.simulation.system.getNumParticles()):
                mass_matrix = np.append(mass_matrix, self.simulation.system.getParticleMass(i).value_in_unit(dalton))
            rpmd_states     = []
            state_positions = []
            for i in range(self.simulation.integrator.getNumCopies()):
                rpmd_state = self.simulation.integrator.getState(i,getPositions=True,getForces=True,getEnergy=False,
                                getParameters=False,enforcePeriodicBox=True,groups=-1)
                rpmd_states.append(rpmd_state)
                state_positions.append(rpmd_state.getPositions().value_in_unit(nanometer))
            rpmd_frame_props['States']    = rpmd_states
            rpmd_frame_props['Positions'] = state_positions
            rpmd_frame_props['T']      = self.simulation.integrator.getTemperature() 
            rpmd_frame_props['P']      = self.simulation.integrator.getNumCopies()
            rpmd_frame_props['N']      = self.simulation.system.getNumParticles()
            rpmd_frame_props['Masses'] = mass_matrix
            rpmd_frame_props['Vsites'] = vsites
        if not traj: 
            Result = evaluate_potential(self.simulation).value_in_unit(kilojoules_per_mole) + \
                        centroid_kinetic(self.simulation, rpmd_frame_props).value_in_unit(kilojoules_per_mole) 
            return Result
        cv_force_terms = []
        potentials = []
        if hasattr(self, 'xyz_rpmd'):
            for I in range(len(self.xyz_rpmd)):                                                                          
                self.set_positions(I) 
                self.rpmd_states = []
                self.state_positions = []
                for i in range(self.simulation.integrator.getNumCopies()):
                    rpmd_state = self.simulation.integrator.getState(i,getPositions=True,getForces=True,getEnergy=False,
                                                                        getParameters=False,enforcePeriodicBox=True,groups=-1)
                    self.rpmd_states.append(rpmd_state)
                    self.state_positions.append(rpmd_state.getPositions().value_in_unit(nanometer))
                self.rpmd_frame_props['States'] = self.rpmd_states
                self.rpmd_frame_props['Positions'] = self.state_positions
                cv_force_terms.append(centroid_kinetic(self.simulation, self.rpmd_frame_props).value_in_unit(kilojoules_per_mole))
                potentials.append(evaluate_potential(self.simulation).value_in_unit(kilojoules_per_mole))
        Result = np.array(cv_force_terms) + np.array(potentials)
        return Result

    def calc_cv(self):
        if not hasattr(self, 'rpmd_frame_props'):
            rpmd_frame_props = {}
            rpmd_frame_props['States'] = []
            vsites = []
            for i in range(self.simulation.system.getNumParticles()):
                if self.simulation.system.isVirtualSite(i):
                    vsites.append(True)
                else:
                    vsites.append(False)
            mass_matrix = np.array([])
            for i in range(self.simulation.system.getNumParticles()):
                mass_matrix = np.append(mass_matrix, self.simulation.system.getParticleMass(i).value_in_unit(dalton))
            rpmd_states     = []
            state_positions = []
            for i in range(self.simulation.integrator.getNumCopies()):
                rpmd_state = self.simulation.integrator.getState(i,getPositions=True,getForces=True,getEnergy=False, 
                                getParameters=False,enforcePeriodicBox=True,groups=-1)
                rpmd_states.append(rpmd_state)
                state_positions.append(rpmd_state.getPositions().value_in_unit(nanometer))
            rpmd_frame_props['States']    = rpmd_states
            rpmd_frame_props['Positions'] = state_positions
            rpmd_frame_props['T']         = self.simulation.integrator.getTemperature()
            rpmd_frame_props['P']         = self.simulation.integrator.getNumCopies()
            rpmd_frame_props['N']         = self.simulation.system.getNumParticles()
            rpmd_frame_props['Masses']    = mass_matrix
            rpmd_frame_props['Vsites']    = vsites
        self.rpmd_states = []
        self.state_positions = []
        for i in range(self.simulation.integrator.getNumCopies()):
            rpmd_state = self.simulation.integrator.getState(i,getPositions=True,getForces=True,getEnergy=False,
                            getParameters=False,enforcePeriodicBox=True,groups=-1)
            self.rpmd_states.append(rpmd_state)
            self.state_positions.append(rpmd_state.getPositions().value_in_unit(nanometer))
            self.rpmd_frame_props['States'] = self.rpmd_states
            self.rpmd_frame_props['Positions'] = self.state_positions
        return centroid_kinetic(self.simulation, self.rpmd_frame_props).value_in_unit(kilojoules_per_mole)

    def evaluate_one_(self, force=False, dipole=False):
        """ Perform a single point calculation on the current geometry. """
        if not hasattr(self, 'xyz_rpmd'): 
            State = self.simulation.context.getState(getPositions=dipole, getEnergy=True, getForces=force)
        Result = {}
        Result["Energy"] = evaluate_potential(self.simulation) / kilojoules_per_mole
        if force: 
            Force = list(np.array(get_forces(self.simulation) / kilojoules_per_mole * nanometer).flatten())
            # Extract forces belonging to real atoms only
            Result["Force"] = np.array(list(itertools.chain(*[Force[3*i:3*i+3] for i in range(len(Force)/3) if self.AtomMask[i]])))
        if dipole:
            State = self.simulation.context.getState(getPositions=True, getEnergy=True, getForces=True)	
            Result["Dipole"] = get_dipole(self.simulation, q=self.nbcharges, mass=self.AtomLists['Mass'], positions=State.getPositions())
        return Result

    def evaluate_(self, force=False, dipole=False, traj=False, rpmd=False):

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
        if not traj: return self.evaluate_one_(force, dipole)
        Energies = []
        Forces = []
        Dipoles = []
        RPMD_CV_est = []
        if hasattr(self, 'xyz_rpmd'):
            for I in range(len(self.xyz_rpmd)):
                self.set_positions(I) 
                R1 = self.evaluate_one_(force,dipole=False)
                Energies.append(R1["Energy"])
                if force: Forces.append(R1["Force"])
                if dipole: Dipoles.append(rpmd_dips(self.simulation, self.nbcharges, self.AtomLists['Mass']))
                if rpmd: RPMD_CV_est.append(self.calc_cv())
        else:
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
        if rpmd: Result["RPMD_CV_est"] = np.array(RPMD_CV_est) + np.array(Energies)
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

    def energy_dipole_rpmd(self):
        Result = self.evaluate_(dipole=True, traj=True, rpmd=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["Dipole"], Result["RPMD_CV_est"].reshape(-1,1)))

    def energy_rpmd(self, rpmd=True):
        Result = self.evaluate_(traj=True, rpmd=True)
        return np.hstack((Result["Energy"].reshape(-1,1), Result["RPMD_CV_est"].reshape(-1,1)))

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
        iequil = int(nequil/nsave)

        if float(int(float(nsteps)/float(nsave))) != float(nsteps)/float(nsave):
            logger.error("Please set nsteps to an integer multiple of nsave\n")
            raise RuntimeError
        isteps = int(nsteps/nsave)
        nsave = int(nsave)

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
        # Stored coordinates, box vectors for RPMD and classical
        if not self.rpmd:
            self.xyz_omms = []
        else:
            if not isinstance(self.simulation.integrator, RPMDIntegrator):
                logger.error('Specified an RPMD simulation but the integrator is wrong!')
                raise RuntimeError
            # Structure of xyz_rpmd is [step][i], where index i=0 contains a list of coordinates for each copy
	        # of the system and index i=1 contains box vectors. 
            self.xyz_rpmd = []
        # Densities, potential and kinetic energies, box volumes, dipole moments
        Rhos = []
        Potentials = []
        Kinetics = []
        Primitive_kinetics = []
        Cp_corrections = []
        Volumes = []
        Dips = []
        Temps = []
        self.rpmd_frame_props = {}
        self.rpmd_frame_props['States'] = []
        if self.rpmd:
            vsites = []
            # Build boolean array indicating which particles are virtual
            for i in range(self.simulation.system.getNumParticles()):
                if self.simulation.system.isVirtualSite(i):
                    vsites.append(True)
                else:
                    vsites.append(False)
            # Build np array of particle masses. Strip units to speed up function evaluation.
            mass_matrix = np.array([])
            for i in range(self.simulation.system.getNumParticles()):
                mass_matrix = np.append(mass_matrix, self.simulation.system.getParticleMass(i).value_in_unit(dalton))
            self.rpmd_frame_props['T']      = self.simulation.integrator.getTemperature()
            self.rpmd_frame_props['P']      = self.simulation.integrator.getNumCopies()
            self.rpmd_frame_props['N']      = self.simulation.system.getNumParticles()
            self.rpmd_frame_props['Masses'] = mass_matrix
            self.rpmd_frame_props['Vsites'] = vsites
            if any(vsites):
                Natoms = int( 3 * self.rpmd_frame_props['N'] / 4)
            else:
                Natoms = self.rpmd_frame_props['N']
            P = self.rpmd_frame_props['P']
            k_b = 0.00831446 * nanometer**2*dalton/(picosecond**2*kelvin)
        #========================#
        # Now run the simulation #
        #========================#
        # Initialize velocities.
        #
        self.simulation.context.setVelocitiesToTemperature(temperature*kelvin)
        #print(XmlSerializer.serialize(self.simulation.system)) 
        # Equilibrate.
        if iequil > 0: 
            if verbose: logger.info("Equilibrating...\n")
            if self.pbc:
                if verbose: logger.info("%6s %9s %9s %13s %10s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)", "Vol(nm^3)", "Rho(kg/m^3)"))
            else:
                if verbose: logger.info("%6s %9s %9s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)"))
        for iteration in range(-1 if self.tdiv == 1 else 0, iequil):
            if iteration >= 0:
                self.simulation.step(nsave)
            if not self.rpmd:
                state = self.simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=False,getForces=False)
            else:
                self.rpmd_states     = []
                self.inst_kinetics   = []
                self.state_positions = []
                for i in range(self.simulation.integrator.getNumCopies()):
                    rpmd_state = self.simulation.integrator.getState(i,getPositions=True,getForces=True,getEnergy=True,
                                                                        getParameters=False,enforcePeriodicBox=True,groups=-1)
                    self.rpmd_states.append(rpmd_state)
                    self.inst_kinetics.append(rpmd_state.getKineticEnergy())
                    self.state_positions.append(rpmd_state.getPositions().value_in_unit(nanometer))
                state = self.rpmd_states[0]
                self.rpmd_frame_props['States']        = self.rpmd_states
                self.rpmd_frame_props['Inst_kinetics'] = self.inst_kinetics
                self.rpmd_frame_props['Positions']     = self.state_positions
            if not self.rpmd:
                kinetic=evaluate_kinetic(self.simulation, self.rpmd_frame_props)
            else:
                primitive_kinetic_tuple = evaluate_kinetic(self.simulation, self.rpmd_frame_props)
                kinetic = centroid_kinetic(self.simulation, self.rpmd_frame_props) 
            potential = evaluate_potential(self.simulation)
            if self.pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (self.mass / volume).in_units_of(kilogram / meter**3)
            else:
                volume = 0.0 * nanometers ** 3
                density = 0.0 * kilogram / meter ** 3
            if not self.rpmd:
                kinetic_temperature = 2.0 * kinetic / kB / self.ndof # (1/2) ndof * kB * T = KE
            else:
                kinetic_temperature = sum(self.rpmd_frame_props['Inst_kinetics']) * 2.0 / (3.0 * Natoms * k_b * P**2)
            if self.pbc:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f %10.4f %13.4f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                                     kinetic_temperature / kelvin, potential / kilojoules_per_mole,
                                                                                     volume / nanometers**3, density / (kilogram / meter**3)))
            else:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                       kinetic_temperature / kelvin, potential / kilojoules_per_mole))
        if verbose: logger.info("Production...\n")
        if self.pbc:
            if verbose: logger.info("%6s %9s %9s %13s %10s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)", "Vol(nm^3)", "Rho(kg/m^3)"))
        else:
            if verbose: logger.info("%6s %9s %9s %13s\n" % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)"))
        if save_traj:
            self.simulation.reporters.append(PDBReporter('%s-md.pdb' % self.name, nsteps))
            self.simulation.reporters.append(DCDReporter('%s-md.dcd' % self.name, nsave))
        for iteration in range(-1 if self.tdiv == 1 else 0, isteps):
            # Propagate dynamics.
            if iteration >= 0: self.simulation.step(nsave)
            # Compute properties.
            if not self.rpmd:
                state = self.simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=False,getForces=False)
            else:
                self.rpmd_states     = []
                self.inst_kinetics   = []
                self.state_positions = []
                for i in range(self.simulation.integrator.getNumCopies()):
                    rpmd_state = self.simulation.integrator.getState(i,getPositions=True,getForces=True,getEnergy=True,
                                                                        getParameters=False,enforcePeriodicBox=True,groups=-1)
                    self.rpmd_states.append(rpmd_state)
                    self.inst_kinetics.append(rpmd_state.getKineticEnergy())
                    self.state_positions.append(rpmd_state.getPositions().value_in_unit(nanometer))
                state = self.rpmd_states[0]
                self.rpmd_frame_props['States']        = self.rpmd_states
                self.rpmd_frame_props['Inst_kinetics'] = self.inst_kinetics
                self.rpmd_frame_props['Positions']     = self.state_positions
            if not self.rpmd:
                kinetic=evaluate_kinetic(self.simulation, self.rpmd_frame_props)
            else:
                kinetic = centroid_kinetic(self.simulation, self.rpmd_frame_props)
                primitive_kinetic_tuple = evaluate_kinetic(self.simulation, self.rpmd_frame_props) 
            potential = evaluate_potential(self.simulation)
            if not self.rpmd:
                kinetic_temperature = 2.0 * kinetic / kB / self.ndof
            else:
                kinetic_temperature = sum(self.rpmd_frame_props['Inst_kinetics']) * 2.0 / (3.0 * Natoms * k_b * P**2)
            if self.pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (self.mass / volume).in_units_of(kilogram / meter**3)
            else:
                box_vectors = None
                volume = 0.0 * nanometers ** 3
                density = 0.0 * kilogram / meter ** 3
            if not self.rpmd:
                self.xyz_omms.append([state.getPositions(), box_vectors])
            else:
                self.xyz_rpmd.append([[self.rpmd_states[i].getPositions() for i in range(self.simulation.integrator.getNumCopies())], box_vectors])
            # Perform energy decomposition.
            for comp, val in energy_components(self.simulation).items():
                if comp in edecomp:
                    edecomp[comp].append(val)
                else:
                    edecomp[comp] = [val]
            if self.pbc:
                if verbose: logger.info("%6d %9.3f %9.3f %13.3f %10.4f %13.4f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                                     kinetic_temperature / kelvin, potential / kilojoules_per_mole,
                                                                                     volume / nanometers**3, density / (kilogram / meter**3)))
            else:
                if verbose: logger.info("%6d %9.3f %9.3f % 13.3f\n" % (iteration+1, state.getTime() / picoseconds,
                                                                       kinetic_temperature / kelvin, potential / kilojoules_per_mole))
            Temps.append(kinetic_temperature / kelvin)
            Rhos.append(density.value_in_unit(kilogram / meter**3))
            Potentials.append(potential / kilojoules_per_mole)
            Kinetics.append(kinetic / kilojoules_per_mole)
            if self.rpmd:
                Primitive_kinetics.append(primitive_kinetic_tuple[0] / kilojoules_per_mole)
            Volumes.append(volume / nanometer**3)
            if not self.rpmd:
                Dips.append(get_dipole(self.simulation,positions=self.xyz_omms[-1][0]))
            else:
                temp_dips = []
                for i in range(self.simulation.integrator.getNumCopies()):
                    temp_dips.append(get_dipole(self.simulation, positions=self.xyz_rpmd[-1][0][i]))
                dip_avg = [sum(col) / float(len(col)) for col in zip(*temp_dips)]
                Dips.append(dip_avg)
        if not self.rpmd:
            Kinetics = np.array(Kinetics)
        else:
            # Add all constant terms to energy estimator time series
            T = self.rpmd_frame_props['T']
            P = self.rpmd_frame_props['P']
            kb = 0.00831446 * nanometer**2*dalton/(picosecond**2*kelvin)
            beta = 1.0/(kb*T)
            kT = (kb * T).value_in_unit(kilojoule_per_mole)
            primitive_kinetic_constant = (3.0 * Natoms * P / (2.0 * beta)).value_in_unit(kilojoule_per_mole)
            centroid_kinetic_constant  = (3.0 * Natoms / (2.0 * beta)).value_in_unit(kilojoule_per_mole)
            Kinetics = np.array(Kinetics) + centroid_kinetic_constant
            Primitive_kinetics = np.array(Primitive_kinetics) + primitive_kinetic_constant 
            Cp_corrections = 2 * kT * Primitive_kinetics - 3 * Natoms * P * kT**2 / 2
        Rhos = np.array(Rhos)
        Potentials = np.array(Potentials)
        Volumes = np.array(Volumes)
        Dips = np.array(Dips)
        Ecomps = OrderedDict([(key, np.array(val)) for key, val in edecomp.items()])
        Ecomps["Potential Energy"] = np.array(Potentials)
        Ecomps["Kinetic Energy"] = np.array(Kinetics)
        Ecomps["Temperature"] = np.array(Temps)
        Ecomps["Total Energy"] = np.array(Potentials) + np.array(Kinetics)
        # Initialized property dictionary.
        prop_return = OrderedDict()
        prop_return.update({'Rhos': Rhos, 'Potentials': Potentials, 'Kinetics': Kinetics, 'Volumes': Volumes, 'Dips': Dips, 'Ecomps': Ecomps,
            'Cp_corrections': Cp_corrections, 'Primitive_kinetics':Primitive_kinetics})
        return prop_return

class Liquid_OpenMM(Liquid):
    """ Condensed phase property matching using OpenMM. """
    def __init__(self,options,tgt_opts,forcefield):
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'force_cuda',forceprint=True)
        # Enable multiple timestep integrator
        self.set_option(tgt_opts,'mts_integrator',forceprint=True)
        # Enable ring polymer MD
        self.set_option(options,'rpmd_opts',forceprint=True)
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
        if tgt_opts['remote_backup']:
            self.nptpfx += " -b"
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

class Hydration_OpenMM(Hydration):
    """ Single point hydration free energies using OpenMM. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Default file names for coordinates and key file.
        # self.set_option(tgt_opts,'coords',default="input.pdb")
        self.set_option(tgt_opts,'openmm_precision','precision',default="double", forceprint=True)
        self.set_option(tgt_opts,'openmm_platform','platname',default="CUDA", forceprint=True)
        self.engine_ = OpenMM
        self.engname = "openmm"
        ## Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ['runcuda.sh']
        ## Suffix for coordinate files.
        self.crdsfx = '.pdb'
        ## Command prefix.
        self.prefix = "bash runcuda.sh"
        if tgt_opts['remote_backup']:
            self.prefix += " -b"
        ## Initialize base class.
        super(Hydration_OpenMM,self).__init__(options,tgt_opts,forcefield)
        ## Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output = ['openmm-md.dcd']
