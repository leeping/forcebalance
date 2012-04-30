""" @package openmmio OpenMM input/output.

@author Lee-Ping Wang
@date 04/2012
"""

import os
# from re import match, sub
# from nifty import isint, isfloat
# from numpy import array
from basereader import BaseReader
# from subprocess import Popen, PIPE
from forceenergymatch import ForceEnergyMatch
from propertymatch import PropertyMatch
import numpy as np
import sys
import pickle
import shutil

try:
    from simtk.openmm.app import *
    from simtk.openmm import *
    from simtk.unit import *
    import work_queue
except:
    pass

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

pdict = "XML_Override"

class OpenMM_Reader(BaseReader):
    """Class for parsing OpenMM force field files.
    
    
    """
    
    def __init__(self,fnm):
        # Initialize the superclass. :)
        super(OpenMM_Reader,self).__init__(fnm)
        ## The parameter dictionary (defined in this file)
        self.pdict  = pdict

    def build_pid(self, element, parameter):
        InteractionType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:] + [element.tag])
        Involved = '.'.join([element.attrib[i] for i in suffix_dict[InteractionType]])#suffix_dict[InteractionType]
        return "/".join([InteractionType, parameter, Involved])

class PropertyMatch_OpenMM(PropertyMatch):
    def __init__(self,options,sim_opts,forcefield):
        ## Initialize the SuperClass!
        super(PropertyMatch_OpenMM,self).__init__(options,sim_opts,forcefield)
        work_queue.set_debug_flag('all')
        self.wq = work_queue.WorkQueue(port=9230, exclusive=False, shutdown=True)
        self.wq.specify_name('forcebalance')
        print('THE PORT IS %d' % self.wq.port)

    def prepare_temp_directory(self,options,sim_opts,forcefield):
        abstempdir = os.path.join(self.root,self.tempdir)
        shutil.copy2(os.path.join(self.root,self.simdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))
        shutil.copy2(os.path.join(self.root,self.simdir,"runcuda.sh"),os.path.join(abstempdir,"runcuda.sh"))
        shutil.copy2(os.path.join(self.root,self.simdir,"npt.py"),os.path.join(abstempdir,"npt.py"))

    def queue_up(self, command, input_files, output_files):
        ### This block submits an OpenMM job ###
        # Required: Command to run             #
        # Required: List of input files        #
        # Required: List of output files       #
        ########################################
        #print "Queueing up command:", command
        #print "Input files:", input_files
        #print "Output files:", output_files
        
        task = work_queue.Task(command)
        for f in input_files:
            task.specify_input_file(f[0],f[1])
        for f in output_files:
            task.specify_output_file(f[0],f[1])
        task.specify_algorithm(work_queue.WORK_QUEUE_SCHEDULE_FCFS)
        task.specify_tag(command)
        self.wq.submit(task)

    def execute(self, temperature, run_dir):
        for fnm in os.listdir(os.path.join(self.root,self.tempdir)):
            if os.path.isfile(os.path.join(self.root,self.tempdir,fnm)):
                shutil.copy2(os.path.join(self.root,self.tempdir,fnm),os.path.join(run_dir,fnm))
        self.queue_up(command = './runcuda.sh python npt.py conf.pdb %s %i 1.0 &> npt.out' % (self.FF.fnms[0], temperature),
                      input_files = [(os.path.join(run_dir,'runcuda.sh'),'runcuda.sh'),
                                     (os.path.join(run_dir,'npt.py'),'npt.py'),
                                     (os.path.join(run_dir,'conf.pdb'),'conf.pdb'),
                                     (os.path.join(run_dir,'ff.p'),'ff.p'),
                                     (os.path.join(run_dir,self.FF.fnms[0]),self.FF.fnms[0])],
                      output_files = [(os.path.join(run_dir,'npt.out'),'npt.out'),
                                      (os.path.join(run_dir,'dynamics.dcd'),'dynamics.dcd')])
        


    # def parse_output(self):
    #     # parse the output file for data!
        


class ForceEnergyMatch_OpenMM(ForceEnergyMatch):

    """Subclass of FittingSimulation for force and energy matching
    using OpenMM.  Implements the prepare and energy_force_driver
    methods.  The get method is in the superclass.  """

    def __init__(self,options,sim_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.trajfnm = "all.gro"
        ## Initialize the SuperClass!
        super(ForceEnergyMatch_OpenMM,self).__init__(options,sim_opts,forcefield)

    def prepare_temp_directory(self, options, sim_opts):
        abstempdir = os.path.join(self.root,self.tempdir)
        # Link the necessary programs into the temporary directory
        #os.symlink(os.path.join(options['tinkerpath'],"testgrad"),os.path.join(abstempdir,"testgrad"))
        # Link the run parameter file
        os.symlink(os.path.join(self.root,self.simdir,"conf.pdb"),os.path.join(abstempdir,"conf.pdb"))

    def energy_force_driver_all_external_(self):
        # This line actually runs OpenMM
        o, e = Popen(["./openmm_energy_force.py","conf.pdb","all.gro","amoebawater.xml"],stdout=PIPE,stderr=PIPE).communicate()
        Answer = pickle.load("Answer.dat")
        M = np.array(list(Answer['Energies']) + list(Answer['Forces']))
        print "Mao"
        print M.shape
        return M

    def energy_force_driver_all_internal_(self):
        pdb = PDBFile("conf.pdb")
        forcefield = ForceField(self.FF.fnms[0])
        
        ################################################
        #       Simulation settings (IMPORTANT)        #
        # Agrees with TINKER to within 0.0001 kcal! :) #
        ################################################
        # Use for Mutual
        # system = forcefield.createSystem(pdb.topology,rigidWater=False,mutualInducedTargetEpsilon=1e-6)
        # Use for Direct
        system = forcefield.createSystem(pdb.topology,rigidWater=False,polarization='Direct')

        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
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
        return M

    def energy_observable_dot_internal_(self, O):
        """Returns an array E_i dot O_i.  This is supposed to be part of a finite difference derivative."""

        pdb = PDBFile("conf.pdb")
        forcefield = ForceField(self.FF.fnms[0])
        
        # Use for Mutual
        # system = forcefield.createSystem(pdb.topology,rigidWater=False,mutualInducedTargetEpsilon=1e-6)
        # Use for Direct
        system = forcefield.createSystem(pdb.topology,rigidWater=False,polarization='Direct')
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)

        E = []
        # Loop through the snapshots
        for I in range(self.ns):
            xyz = self.traj.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # Set the positions using the trajectory
            simulation.context.setPositions(xyz_omm)
            # Set the periodic box size
            simulation.context.setPeriodicBoxVectors(self.traj.boxes[I])
            # Compute the potential energy and append to list
            Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
        return np.mean(np.array(E)*O)

    def energy_total_internal_(self, O):
        pdb = PDBFile("conf.pdb")
        forcefield = ForceField(self.FF.fnms[0])
        
        # Use for Mutual
        # system = forcefield.createSystem(pdb.topology,rigidWater=False,mutualInducedTargetEpsilon=1e-6)
        # Use for Direct
        system = forcefield.createSystem(pdb.topology,rigidWater=False,polarization='Direct')
        # Create the simulation; we're not actually going to use the integrator
        integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(pdb.topology, system, integrator)

        E = []
        # Loop through the snapshots
        for I in range(self.ns):
            xyz = self.traj.xyzs[I]
            xyz_omm = [Vec3(i[0],i[1],i[2]) for i in xyz]*angstrom
            # Set the positions using the trajectory
            simulation.context.setPositions(xyz_omm)
            # Set the periodic box size
            simulation.context.setPeriodicBoxVectors(self.traj.boxes[I])
            # Compute the potential energy and append to list
            Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
        return np.dot(np.array(E),O)

    def energy_force_driver_all(self):
        if self.run_internal:
            return self.energy_force_driver_all_internal_()
        else:
            return self.energy_force_driver_all_external_()
