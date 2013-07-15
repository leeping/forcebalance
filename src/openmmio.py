""" @package forcebalance.openmmio OpenMM input/output.

@author Lee-Ping Wang
@date 04/2012
"""

import os
from forcebalance.basereader import BaseReader
from forcebalance.abinitio import AbInitio
from forcebalance.liquid import Liquid
from forcebalance.interaction import Interaction
import numpy as np
import sys
from forcebalance.finite_difference import *
import pickle
import shutil
from forcebalance.molecule import *
from forcebalance.chemistry import *
from forcebalance.nifty import *
from forcebalance.nifty import _exec
from collections import OrderedDict
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import simtk.openmm._openmm as _openmm
except:
    pass

def get_dipole(simulation,q=None,positions=None):
    """Return the current dipole moment in Debye.
    Note that this quantity is meaningless if the system carries a net charge."""
    dx = 0.0
    dy = 0.0
    dz = 0.0
    enm_debye = 48.03204255928332 # Conversion factor from e*nm to Debye
    for i in simulation.system.getForces():
        if i.__class__.__name__ == "AmoebaMultipoleForce":
            mm = i.getSystemMultipoleMoments(simulation.context)
            dx += mm[1]
            dy += mm[2]
            dz += mm[3]
        if i.__class__.__name__ == "NonbondedForce":
            # Get array of charges.
            if q == None:
                q = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])
            # Get array of positions in nanometers.
            if positions == None:
                positions = simulation.context.getState(getPositions=True).getPositions()
            #x = np.array([j._value for j in positions])
            x = np.array(positions.value_in_unit(nanometer))
            # Multiply charges by positions to get dipole moment.
            dip = enm_debye * np.sum(x*q.reshape(-1,1),axis=0)
            dx += dip[0]
            dy += dip[1]
            dz += dip[2]
    return [dx,dy,dz]

def ResetVirtualSites(positions, system):
    """Given a set of OpenMM-compatible positions and a System object,
    compute the correct virtual site positions according to the System."""
    if any([system.isVirtualSite(i) for i in range(system.getNumParticles())]):
        pos = positions.value_in_unit(nanometer)
        for i in range(system.getNumParticles()):
            if system.isVirtualSite(i):
                vs = system.getVirtualSite(i)
                vstype = vs.__class__.__name__
                # if vstype == 'TwoParticleAverageSite':
                #     vspos = vs.getWeight(0)*pos[vs.getParticle(0)] + vs.getWeight(1)*pos[vs.getParticle(1)]
                # elif vstype == 'ThreeParticleAverageSite':
                #     vspos = vs.getWeight(0)*pos[vs.getParticle(0)] + vs.getWeight(1)*pos[vs.getParticle(1)] + vs.getWeight(2)*pos[vs.getParticle(2)]
                # elif vstype == 'OutOfPlaneSite':
                #     v1 = pos[vs.getParticle(1)] - pos[vs.getParticle(0)]
                #     v2 = pos[vs.getParticle(2)] - pos[vs.getParticle(0)]
                #     cross = Vec3(v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0])
                #     vspos = pos[vs.getParticle(0)] + vs.getWeight12()*v1 + vs.getWeight13()*v2 + vs.getWeightCross()*cross
                # Faster code to work around Python API slowness.
                if vstype == 'TwoParticleAverageSite':
                    vspos = _openmm.TwoParticleAverageSite_getWeight(vs, 0)*pos[_openmm.VirtualSite_getParticle(vs, 0)] + _openmm.TwoParticleAverageSite_getWeight(vs, 1)*pos[_openmm.VirtualSite_getParticle(vs, 1)]
                elif vstype == 'ThreeParticleAverageSite':
                    vspos = _openmm.ThreeParticleAverageSite_getWeight(vs, 0)*pos[_openmm.VirtualSite_getParticle(vs, 0)] + _openmm.ThreeParticleAverageSite_getWeight(vs, 1)*pos[_openmm.VirtualSite_getParticle(vs, 1)] + _openmm.ThreeParticleAverageSite_getWeight(vs, 2)*pos[_openmm.VirtualSite_getParticle(vs, 2)]
                elif vstype == 'OutOfPlaneSite':
                    v1 = pos[_openmm.VirtualSite_getParticle(vs, 1)] - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    v2 = pos[_openmm.VirtualSite_getParticle(vs, 2)] - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    cross = Vec3(v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0])
                    vspos = pos[_openmm.VirtualSite_getParticle(vs, 0)] + _openmm.OutOfPlaneSite_getWeight12(vs)*v1 + _openmm.OutOfPlaneSite_getWeight13(vs)*v2 + _openmm.OutOfPlaneSite_getWeightCross(vs)*cross
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
                Involved = '.'.join([element.attrib[i] for i in suffix_dict[ParentType][InteractionType]])
            return "/".join([InteractionType, parameter, Involved])
        except:
            print "Minor warning: Parameter ID %s doesn't contain any atom types, redundancies are possible" % ("/".join([InteractionType, parameter]))
            return "/".join([InteractionType, parameter])

class Liquid_OpenMM(Liquid):
    def __init__(self,options,tgt_opts,forcefield):
        super(Liquid_OpenMM,self).__init__(options,tgt_opts,forcefield)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'force_cuda',forceprint=True)
        # Enable anisotropic periodic box
        self.set_option(tgt_opts,'anisotropic_box',forceprint=True)
        # Enable multiple timestep integrator
        self.set_option(tgt_opts,'mts_vvvr',forceprint=True)
        # Set up for polarization correction
        if self.do_self_pol:
            self.mpdb = PDBFile(os.path.join(self.root,self.tgtdir,"mono.pdb"))
            forcefield = ForceField(os.path.join(self.root,options['ffdir'],self.FF.openmmxml))
            mod = Modeller(self.mpdb.topology, self.mpdb.positions)
            mod.addExtraParticles(forcefield)
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water)
            # Create the simulation; we're not actually going to use the integrator
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            # Create a Reference platform (this will be faster than CUDA since it's small)
            self.platform = openmm.Platform.getPlatformByName('Reference')
            # Create the simulation object
            self.msim = Simulation(mod.topology, system, integrator, self.platform)
        # I shall enable starting simulations for many different initial conditions.
        self.conf_pdb = Molecule(os.path.join(self.root, self.tgtdir,"conf.pdb"))
        self.traj = None
        if os.path.exists(os.path.join(self.root, self.tgtdir,"all.gro")):
            self.traj = Molecule(os.path.join(self.root, self.tgtdir,"all.gro"))
            print "Found collection of starting conformations, length %i!" % len(self.traj)

    def prepare_temp_directory(self,options,tgt_opts):
        """ Prepare the temporary directory by copying in important files. """
        abstempdir = os.path.join(self.root,self.tempdir)
        # LinkFile(os.path.join(self.root,self.tgtdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))
        LinkFile(os.path.join(self.root,self.tgtdir,"mono.pdb"),os.path.join(abstempdir,"mono.pdb"))
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","runcuda.sh"),os.path.join(abstempdir,"runcuda.sh"))
        LinkFile(os.path.join(os.path.split(__file__)[0],"data","npt.py"),os.path.join(abstempdir,"npt.py"))
        #LinkFile(os.path.join(self.root,self.tgtdir,"npt.py"),os.path.join(abstempdir,"npt.py"))

    def polarization_correction(self,mvals):
        self.FF.make(mvals)
        ff = ForceField(self.FF.openmmxml)
        mod = Modeller(self.mpdb.topology, self.mpdb.positions)
        mod.addExtraParticles(ff)
        sys = ff.createSystem(mod.topology, rigidWater=self.FF.rigid_water)
        UpdateSimulationParameters(sys, self.msim)
        self.msim.context.setPositions(mod.getPositions())
        self.msim.minimizeEnergy()
        pos = self.msim.context.getState(getPositions=True).getPositions()
        pos = ResetVirtualSites(pos, sys)
        d = get_dipole(self.msim, positions=pos)
        if not in_fd():
            print "The molecular dipole moment is % .3f debye" % np.linalg.norm(d)
        dd2 = ((np.linalg.norm(d)-self.self_pol_mu0)*debye)**2
        eps0 = 8.854187817620e-12 * coulomb**2 / newton / meter**2
        epol = 0.5*dd2/(self.self_pol_alpha*angstrom**3*4*np.pi*eps0)/(kilojoule_per_mole/AVOGADRO_CONSTANT_NA)
        return epol

    def npt_simulation(self, temperature, pressure, simnum):
        """ Submit a NPT simulation to the Work Queue. """
        wq = getWorkQueue()
        if not (os.path.exists('npt_result.p') or os.path.exists('npt_result.p.bz2')):
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            if self.traj != None:
                self.conf_pdb.xyzs[0] = self.traj.xyzs[simnum%len(self.traj)]
                self.conf_pdb.boxes[0] = self.traj.boxes[simnum%len(self.traj)]
            self.conf_pdb.write('conf.pdb')
            if wq == None:
                print "Running condensed phase simulation locally."
                print "You may tail -f %s/npt.out in another terminal window" % os.getcwd()
                cmdstr = 'bash runcuda.sh python npt.py conf.pdb %s %i %.3f %.3f %.3f %.3f%s%s%s%s%s%s%s%s --liquid_equ_steps %i &> npt.out' % \
                    (self.FF.openmmxml, self.liquid_prod_steps, self.liquid_timestep, 
                     self.liquid_interval, temperature, pressure, 
                     " --force_cuda" if self.force_cuda else "", 
                     " --anisotropic" if self.anisotropic_box else "", 
                     " --mts_vvvr" if self.mts_vvvr else "", 
                     " --minimize_energy" if self.minimize_energy else "", 
                     " --gas_equ_steps %i" % self.gas_equ_steps if self.gas_equ_steps > 0 else "", 
                     " --gas_prod_steps %i" % self.gas_prod_steps if self.gas_prod_steps > 0 else "", 
                     " --gas_timestep %f" % self.gas_timestep if self.gas_timestep > 0.0 else "", 
                     " --gas_interval %f" % self.gas_interval if self.gas_interval > 0.0 else "", 
                     self.liquid_equ_steps)
                _exec(cmdstr)
            else:
                queue_up(wq,
                         command = 'bash runcuda.sh python npt.py conf.pdb %s %i %.3f %.3f %.3f %.3f%s%s%s%s%s%s%s%s --liquid_equ_steps %i &> npt.out' % \
                             (self.FF.openmmxml, self.liquid_prod_steps, self.liquid_timestep, 
                              self.liquid_interval, temperature, pressure, 
                              " --force_cuda" if self.force_cuda else "", 
                              " --anisotropic" if self.anisotropic_box else "", 
                              " --mts_vvvr" if self.mts_vvvr else "", 
                              " --minimize_energy" if self.minimize_energy else "", 
                              " --gas_equ_steps %i" % self.gas_equ_steps if self.gas_equ_steps > 0 else "", 
                              " --gas_prod_steps %i" % self.gas_prod_steps if self.gas_prod_steps > 0 else "", 
                              " --gas_timestep %f" % self.gas_timestep if self.gas_timestep > 0.0 else "", 
                              " --gas_interval %f" % self.gas_interval if self.gas_interval > 0.0 else "", 
                              self.liquid_equ_steps),
                         input_files = ['runcuda.sh', 'npt.py', 'conf.pdb', 'mono.pdb', 'forcebalance.p'],
                         #output_files = ['dynamics.dcd', 'npt_result.p', 'npt.out', self.FF.openmmxml])
                         output_files = ['npt_result.p.bz2', 'npt.out', self.FF.openmmxml],
                         tgt=self)

class AbInitio_OpenMM(AbInitio):

    """Subclass of AbInitio for force and energy matching
    using OpenMM.  Implements the prepare and energy_force_driver
    methods.  The get method is in the superclass.  """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.trajfnm = "all.gro"
        ## Initialize the SuperClass!
        super(AbInitio_OpenMM,self).__init__(options,tgt_opts,forcefield)
        try:
            PlatName = 'CUDA'
            ## Set the simulation platform
            print "Setting Platform to", PlatName
            self.platform = openmm.Platform.getPlatformByName(PlatName)
            ## Set the device to the environment variable or zero otherwise
            device = os.environ.get('CUDA_DEVICE',"0")
            print "Setting Device to", device
            self.platform.setPropertyDefaultValue("CudaDeviceIndex", device)
            self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
        except:
            PlatName = 'Reference'
            print "Setting Platform to", PlatName
            self.platform = openmm.Platform.getPlatformByName(PlatName)
            # warn_press_key("Setting Platform failed!  Have you loaded the CUDA environment variables?")
            # self.platform = None
        if PlatName == "CUDA":
            if tgt_opts['openmm_cuda_precision'] != '':
                print "Setting Precision to %s" % tgt_opts['openmm_cuda_precision'].lower()
                try:
                    self.platform.setPropertyDefaultValue("CudaPrecision",tgt_opts['openmm_cuda_precision'].lower())
                except:
                    raise Exception('Unable to set the CUDA precision!')
        ## Create the simulation object within this class itself.
        pdb = PDBFile(os.path.join(self.root,self.tgtdir,"conf.pdb"))
        forcefield = ForceField(os.path.join(self.root,options['ffdir'],self.FF.openmmxml))
        mod = Modeller(pdb.topology, pdb.positions)
        mod.addExtraParticles(forcefield)
        if self.FF.amoeba_pol == 'mutual':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,mutualInducedTargetEpsilon=1e-6)
        elif self.FF.amoeba_pol == 'direct':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,polarization='Direct')
        elif self.FF.amoeba_pol == 'nonpolarizable':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water)
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        self.simulation = Simulation(mod.topology, system, integrator, self.platform)
        # Generate OpenMM-compatible positions
        self.xyz_omms = []
        for I in range(self.ns):
            xyz = self.traj.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # An extra step with adding virtual particles
            mod = Modeller(pdb.topology, xyz_omm)
            mod.addExtraParticles(forcefield)
            # Set the positions using the trajectory
            self.xyz_omms.append(mod.getPositions())

    def read_topology(self):
        # Arthur: Document this.
        pdb = PDBFile(os.path.join(self.root,self.tgtdir,"conf.pdb"))
        mypdb = Molecule(os.path.join(self.root,self.tgtdir,"conf.pdb"))
        self.AtomLists['Mass'] = [PeriodicTable[i] for i in mypdb.elem]
        self.AtomLists['ParticleType'] = ['A' for i in mypdb.elem] # Assume that all particle types are atoms.
        self.AtomLists['ResidueNumber'] = [a.residue.index for a in list(pdb.getTopology().atoms())]
        self.topology_flag = True
        return

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        ## Link the PDB file
        LinkFile(os.path.join(self.root,self.tgtdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))

    def energy_force_driver_all_external_(self):
        ## This line actually runs OpenMM,
        o, e = Popen(["./openmm_energy_force.py","conf.pdb","all.gro",self.FF.openmmxml],stdout=PIPE,stderr=PIPE).communicate()
        Answer = pickle.load("Answer.dat")
        M = np.array(list(Answer['Energies']) + list(Answer['Forces']))
        return M

    def energy_force_driver_all_internal_(self):
        """ Loop through the snapshots and compute the energies and forces using OpenMM."""
        pdb = PDBFile("conf.pdb")
        forcefield = ForceField(self.FF.openmmxml)
        mod = Modeller(pdb.topology, pdb.positions)
        mod.addExtraParticles(forcefield)
        # List to determine which atoms are real. :)
        isAtom = [i.element != None for i in list(mod.topology.atoms())]
        #==============================================#
        #       Simulation settings (IMPORTANT)        #
        # Agrees with TINKER to within 0.0001 kcal! :) #
        #==============================================#
        if self.FF.amoeba_pol == 'mutual':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,mutualInducedTargetEpsilon=1e-6)
        elif self.FF.amoeba_pol == 'direct':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,polarization='Direct')
        elif self.FF.amoeba_pol == 'nonpolarizable':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water)
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        if hasattr(self,'simulation'):
            UpdateSimulationParameters(system, self.simulation)
            simulation = self.simulation
        else:
            if self.platform != None:
                simulation = Simulation(mod.topology, system, integrator, self.platform)
            else:
                simulation = Simulation(mod.topology, system, integrator)
        M = []
        # Loop through the snapshots
        for I in range(self.ns):
            # Right now OpenMM is a bit bugged because I can't copy vsite parameters.
            # if self.FF.rigid_water:
            #     simulation.context.applyConstraints(1e-8)
            # else:
            #     simulation.context.computeVirtualSites()
            # Compute the potential energy and append to list
            xyz_omm = self.xyz_omms[I]
            simulation.context.setPositions(ResetVirtualSites(xyz_omm, system))
            Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
            # Compute the force and append to list
            Force = list(np.array(simulation.context.getState(getForces=True).getForces() / kilojoules_per_mole * nanometer).flatten())
            # Extract forces belonging to real atoms only
            Force1 = list(itertools.chain(*[Force[3*i:3*i+3] for i in range(len(Force)/3) if isAtom[i]]))
            M.append(np.array([Energy] + Force1))
        M = np.array(M)
        return M

    def energy_force_driver_all(self):
        if self.run_internal:
            return self.energy_force_driver_all_internal_()
        else:
            warn_press_key('The energy_force_driver_all_external_ function is deprecated!')
            return self.energy_force_driver_all_external_()

class Interaction_OpenMM(Interaction):

    """Subclass of Target for interaction matching using OpenMM. """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory file containing snapshots.
        self.trajfnm = "all.pdb"
        ## Dictionary of simulation objects (dimer, fraga, fragb)
        self.simulations = OrderedDict()
        ## Initialize base class.
        super(Interaction_OpenMM,self).__init__(options,tgt_opts,forcefield)

    def prepare_temp_directory(self, options, tgt_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        cwd = os.getcwd()
        os.chdir(abstempdir)
        ## Set up three OpenMM System objects.
        self.traj[0].write("dimer.pdb")
        self.traj[0].atom_select(self.select1).write("fraga.pdb")
        self.traj[0].atom_select(self.select2).write("fragb.pdb")
        # ## Write a single frame PDB if it doesn't exist already. Breaking my self-imposed rule of not editing the Target directory...
        # if not os.path.exists(os.path.join(self.root,self.tgtdir,"conf.pdb")):
        #     self.traj[0].write(os.path.join(self.root,self.tgtdir,"conf.pdb"))
        ## TODO: The following code should not be repeated everywhere.
        for pdbfnm in ["dimer.pdb", "fraga.pdb", "fragb.pdb"]:
            print "Setting up Simulation object for %s" % pdbfnm
            try:
                PlatName = 'CUDA'
                ## Set the simulation platform
                print "Setting Platform to", PlatName
                self.platform = openmm.Platform.getPlatformByName(PlatName)
                ## Set the device to the environment variable or zero otherwise
                device = os.environ.get('CUDA_DEVICE',"0")
                print "Setting Device to", device
                self.platform.setPropertyDefaultValue("CudaDeviceIndex", device)
                self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
            except:
                PlatName = 'Reference'
                print "Setting Platform to", PlatName
                self.platform = openmm.Platform.getPlatformByName(PlatName)
                # warn_press_key("Setting Platform failed!  Have you loaded the CUDA environment variables?")
                # self.platform = None
            if PlatName == "CUDA":
                if tgt_opts['openmm_cuda_precision'] != '':
                    print "Setting Precision to %s" % tgt_opts['openmm_cuda_precision'].lower()
                    try:
                        self.platform.setPropertyDefaultValue("CudaPrecision",tgt_opts['openmm_cuda_precision'].lower())
                    except:
                        raise Exception('Unable to set the CUDA precision!')
            ## Create the simulation object within this class itself.
            pdb = PDBFile(pdbfnm)
            forcefield = ForceField(os.path.join(self.root,options['ffdir'],self.FF.openmmxml))
            mod = Modeller(pdb.topology, pdb.positions)
            mod.addExtraParticles(forcefield)
            if self.FF.amoeba_pol == 'mutual':
                system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,mutualInducedTargetEpsilon=1e-6)
            elif self.FF.amoeba_pol == 'direct':
                system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,polarization='Direct')
            elif self.FF.amoeba_pol == 'nonpolarizable':
                system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water)
            # Create the simulation; we're not actually going to use the integrator
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            self.simulations[os.path.splitext(pdbfnm)[0]] = Simulation(mod.topology, system, integrator, self.platform)
        os.chdir(cwd)
        
    def energy_driver_all(self, mode):
        if mode not in ['dimer','fraga','fragb']:
            raise Exception('This function may only be called with three modes - dimer, fraga, fragb')
        if mode == 'dimer':
            self.traj.write("shot.pdb")
        elif mode == 'fraga':
            self.traj.atom_select(self.select1).write("shot.pdb")
        elif mode == 'fragb':
            self.traj.atom_select(self.select2).write("shot.pdb")
        thistraj = Molecule("shot.pdb")
        
        # Run OpenMM.
        pdb = PDBFile(mode+".pdb")
        forcefield = ForceField(self.FF.openmmxml)
        mod = Modeller(pdb.topology, pdb.positions)
        mod.addExtraParticles(forcefield)
        #==============================================#
        #       Simulation settings (IMPORTANT)        #
        # Agrees with TINKER to within 0.0001 kcal! :) #
        #==============================================#
        if self.FF.amoeba_pol == 'mutual':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,mutualInducedTargetEpsilon=1e-6)
        elif self.FF.amoeba_pol == 'direct':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water,polarization='Direct')
        elif self.FF.amoeba_pol == 'nonpolarizable':
            system = forcefield.createSystem(mod.topology,rigidWater=self.FF.rigid_water)
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        if hasattr(self,'simulations'):
            UpdateSimulationParameters(system, self.simulations[mode])
            simulation = self.simulations[mode]
        else:
            raise Exception('This module only works if it contains a dictionary of Simulation objects.')

        M = []
        # Loop through the snapshots
        for I in range(self.ns):
            xyz = thistraj.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # An extra step with adding virtual particles
            mod = Modeller(pdb.topology, xyz_omm)
            mod.addExtraParticles(forcefield)
            # Set the positions using the trajectory
            simulation.context.setPositions(mod.getPositions())
            # Right now OpenMM is a bit bugged because I can't copy vsite parameters.
            # if self.FF.rigid_water:
            #     simulation.context.applyConstraints(1e-8)
            # Compute the potential energy and append to list
            Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
            M.append(Energy)
        M = np.array(M)
        return M
    
    def interaction_driver_all(self,dielectric=False):
        # Compute the energies for the dimer
        D = self.energy_driver_all('dimer')
        A = self.energy_driver_all('fraga')
        B = self.energy_driver_all('fragb')
        return D - A - B

