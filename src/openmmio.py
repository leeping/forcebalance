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
from nifty import *
try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
except:
    pass

## Dictionary for building parameter identifiers.  As usual they go like this:
## HarmonicBondForce.Bond_length_OW.HW
suffix_dict = { "HarmonicBondForce.Bond" : ["class1","class2"],
                "HarmonicAngleForce.Angle" : ["class1","class2","class3"], 
                "NonbondedForce" : [],
                "NonbondedForce.Atom": ["type"],
                "AmoebaHarmonicBondForce.Bond" : ["class1","class2"],
                "AmoebaHarmonicAngleForce.Angle" : ["class1","class2","class3"],
                "AmoebaVdwForce.Vdw" : ["class"],
                "AmoebaMultipoleForce.Multipole" : ["type","kz","kx"],
                "AmoebaMultipoleForce.Multipole" : ["type","kz","kx"],
                "AmoebaMultipoleForce.Polarize" : ["type"],
                "AmoebaUreyBradleyForce.UreyBradley" : ["class1","class2","class3"]
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
        InteractionType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:] + [element.tag])
        Involved = '.'.join([element.attrib[i] for i in suffix_dict[InteractionType]])#suffix_dict[InteractionType]
        return "/".join([InteractionType, parameter, Involved])

class Liquid_OpenMM(Liquid):
    def __init__(self,options,sim_opts,forcefield):
        super(Liquid_OpenMM,self).__init__(options,sim_opts,forcefield)

    def prepare_temp_directory(self,options,sim_opts):
        """ Prepare the temporary directory by copying in important files. """
        abstempdir = os.path.join(self.root,self.tempdir)
        os.symlink(os.path.join(self.root,self.simdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))
        os.symlink(os.path.join(self.root,self.simdir,"mono.pdb"),os.path.join(abstempdir,"mono.pdb"))
        os.symlink(os.path.join(self.root,self.simdir,"settings","runcuda.sh"),os.path.join(abstempdir,"runcuda.sh"))
        os.symlink(os.path.join(self.root,self.simdir,"settings","npt.py"),os.path.join(abstempdir,"npt.py"))
        os.symlink(os.path.join(self.root,self.simdir,"settings","evaltraj.py"),os.path.join(abstempdir,"evaltraj.py"))

    def npt_simulation(self, temperature):
        """ Submit a NPT simulation to the Work Queue. """
        link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
        queue_up(self.wq,
                 command = './runcuda.sh python npt.py conf.pdb %s %.1f 1.0 &> npt.out' % (self.FF.fnms[0], temperature),
                 input_files = ['runcuda.sh', 'npt.py', 'conf.pdb', 'mono.pdb', 'forcebalance.p'],
                 output_files = ['dynamics.dcd', 'npt_result.p', 'npt.out', self.FF.fnms[0]])
                 #output_files = ['npt_result.p', 'npt.out', self.FF.fnms[0]])

    def evaluate_trajectory(self, name, trajpath, mvals, bGradient):
        """ Submit an energy / gradient evaluation (looping over a trajectory) to the Work Queue. """
        cwd = os.getcwd()
        rnd = os.path.join(cwd,name)
        os.makedirs(name)
        link_dir_contents(os.path.join(self.root,self.rundir),rnd)
        infnm = os.path.join(rnd,'forcebalance.p')
        os.remove(infnm)
        with open(os.path.join(rnd,'forcebalance.p'),'w') as f: lp_dump((self.FF,mvals,self.h),f)
        queue_up_src_dest(self.wq, command = './runcuda.sh python evaltraj.py conf.pdb %s dynamics.dcd %s &> evaltraj.log' % (self.FF.fnms[0], "True" if bGradient else "False"),
                          input_files = [(os.path.join(rnd,'runcuda.sh'),'runcuda.sh'), 
                                         (os.path.join(rnd,'evaltraj.py'),'evaltraj.py'),
                                         (os.path.join(rnd,'conf.pdb'),'conf.pdb'),
                                         (os.path.join(rnd,'forcebalance.p'),'forcebalance.p'),
                                         (os.path.join(trajpath,'dynamics.dcd'), 'dynamics.dcd')],
                          output_files = [(os.path.join(rnd,'evaltraj_result.p'),'evaltraj_result.p'), 
                                          (os.path.join(rnd,'evaltraj.log'),'evaltraj.log')])
        #wq_wait(self.wq)

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

    def __init__(self,options,sim_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.trajfnm = "all.gro"
        ## Initialize the SuperClass!
        super(AbInitio_OpenMM,self).__init__(options,sim_opts,forcefield)
        try:
            ## Copied over from npt.py (for now)
            PlatName = 'Cuda'
            print "Setting Platform to", PlatName
            self.platform = openmm.Platform.getPlatformByName(PlatName)
            ## Set the device to the environment variable or zero otherwise
            device = os.environ.get('CUDA_DEVICE',"0")
            print "Setting Device to", device
            self.platform.setPropertyDefaultValue("CudaDevice", device)
            self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
        except:
            self.platform = None

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        ## Link the PDB file
        os.symlink(os.path.join(self.root,self.simdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))

    def energy_force_driver_all_external_(self):
        ## This line actually runs OpenMM,
        o, e = Popen(["./openmm_energy_force.py","conf.pdb","all.gro",self.FF.fnms[0]],stdout=PIPE,stderr=PIPE).communicate()
        Answer = pickle.load("Answer.dat")
        M = np.array(list(Answer['Energies']) + list(Answer['Forces']))
        return M

    def energy_force_driver_all_internal_(self):
        """ Loop through the snapshots and compute the energies and forces using OpenMM."""
        pdb = PDBFile("conf.pdb")
        forcefield = ForceField(self.FF.fnms[0])
        #==============================================#
        #       Simulation settings (IMPORTANT)        #
        # Agrees with TINKER to within 0.0001 kcal! :) #
        #==============================================#
        if self.FF.amoeba_pol == 'mutual':
            system = forcefield.createSystem(pdb.topology,rigidWater=False,mutualInducedTargetEpsilon=1e-6)
        elif self.FF.amoeba_pol == 'direct':
            system = forcefield.createSystem(pdb.topology,rigidWater=False,polarization='Direct')
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
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
            return self.energy_force_driver_all_external_()
