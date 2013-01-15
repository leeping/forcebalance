""" @package openmmio OpenMM input/output.

@author Lee-Ping Wang
@date 04/2012
"""

import os
from basereader import BaseReader
from abinitio import AbInitio
from liquid import Liquid
import numpy as np
import sys
from finite_difference import *
import pickle
import shutil
from molecule import *
from chemistry import *
from nifty import *
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
except:
    pass

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

def CopyNonbondedParameters(src, dest):
    dest.setReactionFieldDielectric(src.getReactionFieldDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i,*src.getParticleParameters(i))
    for i in range(src.getNumExceptions()):
        dest.setExceptionParameters(i,*src.getExceptionParameters(i))

def do_nothing(src, dest):
    return

def CopySystemParameters(src,dest):
    # Copy parameters from one system (i.e. that which is created by a new force field)
    # to another system (i.e. the one stored inside the Target object).
    # DANGER: These need to be implemented manually!!!
    Copiers = {'AmoebaBondForce':CopyAmoebaBondParameters,
               'AmoebaOutOfPlaneBendForce':CopyAmoebaOutOfPlaneBendParameters,
               'AmoebaAngleForce':CopyAmoebaAngleParameters,
               'AmoebaInPlaneAngleForce':CopyAmoebaInPlaneAngleParameters,
               'AmoebaVdwForce':CopyAmoebaVdwParameters,
               'AmoebaMultipoleForce':CopyAmoebaMultipoleParameters,
               'HarmonicBondForce':CopyHarmonicBondParameters,
               'HarmonicAngleForce':CopyHarmonicAngleParameters,
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


## Dictionary for building parameter identifiers.  As usual they go like this:
## Bond/length/OW.HW
## The dictionary is two-layered because the same interaction type (Bond)
## could be under two different parent types (HarmonicBondForce, AmoebaHarmonicBondForce)
suffix_dict = { "HarmonicBondForce" : {"Bond" : ["class1","class2"]},
                "HarmonicAngleForce" : {"Angle" : ["class1","class2","class3"],},
                "NonbondedForce" : {"Atom": ["type"]},
                "AmoebaBondForce" : {"Bond" : ["class1","class2"]},
                "AmoebaAngleForce" : {"Angle" : ["class1","class2","class3"]},
                "AmoebaStretchBendForce" : {"StretchBend" : ["class1","class2","class3"]},
                "AmoebaVdwForce" : {"Vdw" : ["class"]},
                "AmoebaMultipoleForce" : {"Multipole" : ["type","kz","kx"], "Polarize" : ["type"]},
                "AmoebaUreyBradleyForce" : {"UreyBradley" : ["class1","class2","class3"]}
                }

## pdict is a useless variable if the force field is XML.
pdict = "XML_Override"

def liquid_energy_driver(mvals,pdb,FF,xyzs,settings,platform,boxes=None,verbose=False):

   """
   Compute a set of snapshot energies as a function of the force field parameters.

   This is a combined OpenMM and ForceBalance function.  Note (importantly) that this
   function creates a new force field XML file in the run directory.

   ForceBalance creates the force field, OpenMM reads it in, and we loop through the snapshots
   to compute the energies.
   
   @todo I should be able to generate the OpenMM force field object without writing an external file.
   @todo This is a sufficiently general function to be merged into openmmio.py?
   @param[in] mvals Mathematical parameter values
   @param[in] pdb OpenMM PDB object
   @param[in] FF ForceBalance force field object
   @param[in] xyzs List of OpenMM positions
   @param[in] settings OpenMM settings for creating the System
   @param[in] boxes Periodic box vectors
   @return E A numpy array of energies in kilojoules per mole

   """
   # Print the force field XML from the ForceBalance object, with modified parameters.
   FF.make(mvals)
   # Load the force field XML file to make the OpenMM object.
   forcefield = ForceField(sys.argv[2])
   # Create the system, setup the simulation.
   system = forcefield.createSystem(pdb.topology, **settings)
   integrator = openmm.LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
   simulation = Simulation(pdb.topology, system, integrator, platform)
   E = []
   # Loop through the snapshots
   if boxes == None:
       for xyz in xyzs:
          # Set the positions and the box vectors
          simulation.context.setPositions(xyz)
          # Compute the potential energy and append to list
          Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
          E.append(Energy)
   else:
       for xyz,box in zip(xyzs,boxes):
          # Set the positions and the box vectors
          simulation.context.setPositions(xyz)
          simulation.context.setPeriodicBoxVectors(box[0],box[1],box[2])
          # Compute the potential energy and append to list
          Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
          E.append(Energy)
   print "\r",
   if verbose: print E
   return np.array(E)

def liquid_energy_derivatives(mvals,h,pdb,FF,xyzs,settings,platform,boxes=None):

   """
   Compute the first and second derivatives of a set of snapshot
   energies with respect to the force field parameters.

   This basically calls the finite difference subroutine on the
   energy_driver subroutine also in this script.

   @todo This is a sufficiently general function to be merged into openmmio.py?
   @param[in] mvals Mathematical parameter values
   @param[in] pdb OpenMM PDB object
   @param[in] FF ForceBalance force field object
   @param[in] xyzs List of OpenMM positions
   @param[in] settings OpenMM settings for creating the System
   @param[in] boxes Periodic box vectors
   @return G First derivative of the energies in a N_param x N_coord array
   @return Hd Second derivative of the energies (i.e. diagonal Hessian elements) in a N_param x N_coord array

   """

   G        = np.zeros((FF.np,len(xyzs)))
   Hd       = np.zeros((FF.np,len(xyzs)))
   E0       = liquid_energy_driver(mvals, pdb, FF, xyzs, settings, platform, boxes)
   for i in range(FF.np):
      G[i,:], Hd[i,:] = f12d3p(fdwrap(liquid_energy_driver,mvals,i,key=None,pdb=pdb,FF=FF,xyzs=xyzs,settings=settings,platform=platform,boxes=boxes),h,f0=E0)
   return G, Hd

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
            Involved = '.'.join([element.attrib[i] for i in suffix_dict[ParentType][InteractionType]])
            return "/".join([InteractionType, parameter, Involved])
        except:
            print "Minor warning: Parameter ID %s doesn't contain any atom types, redundancies are possible" % ("/".join([InteractionType, parameter]))
            return "/".join([InteractionType, parameter])

class Liquid_OpenMM(Liquid):
    def __init__(self,options,tgt_opts,forcefield):
        super(Liquid_OpenMM,self).__init__(options,tgt_opts,forcefield)
        if options['openmm_new_cuda']:
            self.new_cuda = True

    def prepare_temp_directory(self,options,tgt_opts):
        """ Prepare the temporary directory by copying in important files. """
        abstempdir = os.path.join(self.root,self.tempdir)
        os.symlink(os.path.join(self.root,self.tgtdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))
        os.symlink(os.path.join(self.root,self.tgtdir,"mono.pdb"),os.path.join(abstempdir,"mono.pdb"))
        os.symlink(os.path.join(self.root,self.tgtdir,"runcuda.sh"),os.path.join(abstempdir,"runcuda.sh"))
        os.symlink(os.path.join(self.root,self.tgtdir,"npt.py"),os.path.join(abstempdir,"npt.py"))
        os.symlink(os.path.join(self.root,self.tgtdir,"evaltraj.py"),os.path.join(abstempdir,"evaltraj.py"))

    def npt_simulation(self, temperature, pressure):
        """ Submit a NPT simulation to the Work Queue. """
        wq = getWorkQueue()
        if not os.path.exists('npt_result.p'):
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            queue_up(wq,
                     command = './runcuda.sh python npt.py conf.pdb %s %.3f %.3f &> npt.out' % (self.FF.openmmxml, temperature, pressure),
                     input_files = ['runcuda.sh', 'npt.py', 'conf.pdb', 'mono.pdb', 'forcebalance.p'],
                     #output_files = ['dynamics.dcd', 'npt_result.p', 'npt.out', self.FF.openmmxml])
                     output_files = ['npt_result.p', 'npt.out', self.FF.openmmxml])

    def evaluate_trajectory(self, name, trajpath, mvals, bGradient):
        """ Submit an energy / gradient evaluation (looping over a trajectory) to the Work Queue. """
        cwd = os.getcwd()
        rnd = os.path.join(cwd,name)
        os.makedirs(name)
        link_dir_contents(os.path.join(self.root,self.rundir),rnd)
        infnm = os.path.join(rnd,'forcebalance.p')
        os.remove(infnm)
        with open(os.path.join(rnd,'forcebalance.p'),'w') as f: lp_dump((self.FF,mvals,self.h,True),f)
        wq = getWorkQueue()
        queue_up_src_dest(wq, command = './runcuda.sh python evaltraj.py conf.pdb %s dynamics.dcd %s &> evaltraj.log' % (self.FF.openmmxml, "True" if bGradient else "False"),
                          input_files = [(os.path.join(rnd,'runcuda.sh'),'runcuda.sh'), 
                                         (os.path.join(rnd,'evaltraj.py'),'evaltraj.py'),
                                         (os.path.join(rnd,'conf.pdb'),'conf.pdb'),
                                         (os.path.join(rnd,'forcebalance.p'),'forcebalance.p'),
                                         (os.path.join(trajpath,'dynamics.dcd'), 'dynamics.dcd')],
                          output_files = [(os.path.join(rnd,'evaltraj_result.p'),'evaltraj_result.p'), 
                                          (os.path.join(rnd,'evaltraj.log'),'evaltraj.log')])

    def get_evaltraj_result(self, Dict, name, key, bGradient):
        cwd = os.getcwd()
        rnd = os.path.join(cwd,name)
        if bGradient:
            Answer = lp_load(open(os.path.join(rnd,'evaltraj_result.p')))[1]
        else:
            Answer = lp_load(open(os.path.join(rnd,'evaltraj_result.p')))[0]
        Dict[key] = Answer
        
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
            ## Copied over from npt.py (for now)
            if options['openmm_new_cuda']:
                PlatName = 'CUDA'
            else:
                PlatName = 'Cuda'
            ## Set the simulation platform
            print "Setting Platform to", PlatName
            self.platform = openmm.Platform.getPlatformByName(PlatName)
            ## Set the device to the environment variable or zero otherwise
            device = os.environ.get('CUDA_DEVICE',"0")
            print "Setting Device to", device
            self.platform.setPropertyDefaultValue("CudaDevice", device)
            self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
        except:
            warn_press_key("Setting Platform failed!  Have you loaded the CUDA environment variables?")
            self.platform = None
        ## If using the new CUDA platform, then create the simulation object within this class itself.
        if PlatName == "CUDA":
            # Set up the entire system here on the new CUDA Platform.
            pdb = PDBFile(os.path.join(self.root,self.tgtdir,"conf.pdb"))
            forcefield = ForceField(os.path.join(self.root,options['ffdir'],self.FF.openmmxml))
            if self.FF.amoeba_pol == 'mutual':
                system = forcefield.createSystem(pdb.topology,rigidWater=self.FF.rigid_water,mutualInducedTargetEpsilon=1e-6)
            elif self.FF.amoeba_pol == 'direct':
                system = forcefield.createSystem(pdb.topology,rigidWater=self.FF.rigid_water,polarization='Direct')
            # Create the simulation; we're not actually going to use the integrator
            integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
            if self.platform != None:
                self.simulation = Simulation(pdb.topology, system, integrator, self.platform)
            else:
                raise Exception('Unable to set the Platform to CUDA!')

    def read_topology(self):
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
        os.symlink(os.path.join(self.root,self.tgtdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))

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
        #==============================================#
        #       Simulation settings (IMPORTANT)        #
        # Agrees with TINKER to within 0.0001 kcal! :) #
        #==============================================#
        if self.FF.amoeba_pol == 'mutual':
            system = forcefield.createSystem(pdb.topology,rigidWater=self.FF.rigid_water,mutualInducedTargetEpsilon=1e-6)
        elif self.FF.amoeba_pol == 'direct':
            system = forcefield.createSystem(pdb.topology,rigidWater=self.FF.rigid_water,polarization='Direct')
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        if hasattr(self,'simulation'):
            UpdateSimulationParameters(system, self.simulation)
            simulation = self.simulation
        else:
            if self.platform != None:
                simulation = Simulation(pdb.topology, system, integrator, self.platform)
            else:
                simulation = Simulation(pdb.topology, system, integrator)

        M = []
        # Loop through the snapshots
        for I in range(self.ns):
            xyz = self.traj.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # Set the positions using the trajectory
            simulation.context.setPositions(xyz_omm)
            if self.FF.rigid_water:
                simulation.context.applyConstraints(1e-8)
            # Compute the potential energy and append to list
            Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
            # Compute the force and append to list
            Force = list(np.array(simulation.context.getState(getForces=True).getForces() / kilojoules_per_mole * nanometer).flatten())
            M.append(np.array([Energy] + Force))
        M = np.array(M)
        return M

    def energy_force_driver_all(self):
        if self.run_internal:
            return self.energy_force_driver_all_internal_()
        else:
            warn_press_key('The energy_force_driver_all_external_ function is deprecated!')
            return self.energy_force_driver_all_external_()
