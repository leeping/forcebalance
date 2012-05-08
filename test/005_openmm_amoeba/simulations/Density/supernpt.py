#!/usr/bin/env python

"""
@package supernpt

NPT simulation in OpenMM.  Runs a simulation to compute bulk properties
(for example, the density or the enthalpy of vaporization) and compute the
derivative with respect to changing the force field parameters.  

The basic idea is this: First we run a density simulation to determine
the average density.  This quantity of course has some uncertainty,
and in general we want to avoid evaluating finite-difference
derivatives of noisy quantities.  The key is to realize that the
densities are sampled from a Boltzmann distribution, so the analytic
derivative can be computed if the potential energy derivative is
accessible.  We compute the potential energy derivative using
finite-difference of snapshot energies and apply a simple formula to
compute the density derivative.

The enthalpy of vaporization should come just as easily.

This script borrows from John Chodera's ideal gas simulation in PyOpenMM.

References

[1] Shirts MR, Mobley DL, Chodera JD, and Pande VS. Accurate and efficient corrections for
missing dispersion interactions in molecular simulations. JPC B 111:13052, 2007.

[2] Ahn S and Fessler JA. Standard errors of mean, variance, and standard deviation estimators.
Technical Report, EECS Department, The University of Michigan, 2003.

Copyright And License

@author Lee-Ping Wang <leeping@stanford.edu>
@author John D. Chodera <jchodera@gmail.com>

All code in this repository is released under the GNU General Public License.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness for a
particular purpose.  See the GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

"""

#================#
# Global Imports #
#================#

import os
import os.path
import sys
import math
import numpy as np
import simtk
import simtk.unit as units
import simtk.openmm as openmm
from simtk.openmm.app import *
from forcebalance.forcefield import FF
from forcebalance.nifty import col, lp_load
from forcebalance.finite_difference import fdwrap, f12d3p

#======================================================#
# Global, user-tunable variables (simulation settings) #
#======================================================#

# Select run parameters
timestep = 0.5 * units.femtosecond # timestep for integrtion
nsteps = 200                       # number of steps per data record
nequiliterations = 100             # number of equilibration iterations
niterations = 100                  # number of iterations to collect data for

# Set temperature, pressure, and collision rate for stochastic thermostats.
temperature = float(sys.argv[3]) * units.kelvin
pressure = float(sys.argv[4]) * units.atmospheres 
collision_frequency = 91.0 / units.picosecond 
barostat_frequency = 25            # number of steps between MC volume adjustments
nprint = 1

# Flag to set verbose debug output
verbose = True

mutual_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*units.nanometer, 
                 'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 1.2, 
                 'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                 'mutualInducedTargetEpsilon' : 1e-6}

direct_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*units.nanometer, 
                 'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 1.2, 
                 'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                 'polarization' : 'direct'}

def generateMaxwellBoltzmannVelocities(system, temperature):
   """ Generate velocities from a Maxwell-Boltzmann distribution. """
   # Get number of atoms
   natoms = system.getNumParticles()
   # Create storage for velocities.        
   velocities = units.Quantity(np.zeros([natoms, 3], np.float32), units.nanometer / units.picosecond) # velocities[i,k] is the kth component of the velocity of atom i
   # Compute thermal energy and inverse temperature from specified temperature.
   kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
   kT = kB * temperature # thermal energy
   beta = 1.0 / kT # inverse temperature
   # Assign velocities from the Maxwell-Boltzmann distribution.
   for atom_index in range(natoms):
      mass = system.getParticleMass(atom_index) # atomic mass
      sigma = units.sqrt(kT / mass) # standard deviation of velocity distribution for each coordinate for this atom
      for k in range(3):
         velocities[atom_index,k] = sigma * np.random.normal()
   return velocities
   
def statisticalInefficiency(A_n, B_n=None, fast=False, mintime=3):

  """
  Compute the (cross) statistical inefficiency of (two) timeseries.

  Notes 
    The same timeseries can be used for both A_n and B_n to get the autocorrelation statistical inefficiency.
    The fast method described in Ref [1] is used to compute g.

  References  
    [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the weighted
    histogram analysis method for the analysis of simulated and parallel tempering simulations.
    JCTC 3(1):26-41, 2007.

  Examples

  Compute statistical inefficiency of timeseries data with known correlation time.  

  >>> import timeseries
  >>> A_n = timeseries.generateCorrelatedTimeseries(N=100000, tau=5.0)
  >>> g = statisticalInefficiency(A_n, fast=True)

  @param[in] A_n (required, numpy array) - A_n[n] is nth value of
  timeseries A.  Length is deduced from vector.

  @param[in] B_n (optional, numpy array) - B_n[n] is nth value of
  timeseries B.  Length is deduced from vector.  If supplied, the
  cross-correlation of timeseries A and B will be estimated instead of
  the autocorrelation of timeseries A.

  @param[in] fast (optional, boolean) - if True, will use faster (but
  less accurate) method to estimate correlation time, described in
  Ref. [1] (default: False)

  @param[in] mintime (optional, int) - minimum amount of correlation
  function to compute (default: 3) The algorithm terminates after
  computing the correlation time out to mintime when the correlation
  function furst goes negative.  Note that this time may need to be
  increased if there is a strong initial negative peak in the
  correlation function.

  @return g The estimated statistical inefficiency (equal to 1 + 2
  tau, where tau is the correlation time).  We enforce g >= 1.0.
  
  """

  # Create numpy copies of input arguments.
  A_n = np.array(A_n)
  if B_n is not None:  
    B_n = np.array(B_n)
  else:
    B_n = np.array(A_n) 
  # Get the length of the timeseries.
  N = A_n.size
  # Be sure A_n and B_n have the same dimensions.
  if(A_n.shape != B_n.shape):
    raise ParameterError('A_n and B_n must have same dimensions.')
  # Initialize statistical inefficiency estimate with uncorrelated value.
  g = 1.0
  # Compute mean of each timeseries.
  mu_A = A_n.mean()
  mu_B = B_n.mean()
  # Make temporary copies of fluctuation from mean.
  dA_n = A_n.astype(np.float64) - mu_A
  dB_n = B_n.astype(np.float64) - mu_B
  # Compute estimator of covariance of (A,B) using estimator that will ensure C(0) = 1.
  sigma2_AB = (dA_n * dB_n).mean() # standard estimator to ensure C(0) = 1
  # Trap the case where this covariance is zero, and we cannot proceed.
  if(sigma2_AB == 0):
    raise ParameterException('Sample covariance sigma_AB^2 = 0 -- cannot compute statistical inefficiency')
  # Accumulate the integrated correlation time by computing the normalized correlation time at
  # increasing values of t.  Stop accumulating if the correlation function goes negative, since
  # this is unlikely to occur unless the correlation function has decayed to the point where it
  # is dominated by noise and indistinguishable from zero.
  t = 1
  increment = 1
  while (t < N-1):
    # compute normalized fluctuation correlation function at time t
    C = sum( dA_n[0:(N-t)]*dB_n[t:N] + dB_n[0:(N-t)]*dA_n[t:N] ) / (2.0 * float(N-t) * sigma2_AB)
    # Terminate if the correlation function has crossed zero and we've computed the correlation
    # function at least out to 'mintime'.
    if (C <= 0.0) and (t > mintime):
      break
    # Accumulate contribution to the statistical inefficiency.
    g += 2.0 * C * (1.0 - float(t)/float(N)) * float(increment)
    # Increment t and the amount by which we increment t.
    t += increment
    # Increase the interval if "fast mode" is on.
    if fast: increment += 1
  # g must be at least unity
  if (g < 1.0): g = 1.0
  # Return the computed statistical inefficiency.
  return g

def compute_volume(box_vectors):
   """ Compute the total volume of an OpenMM system. """
   [a,b,c] = box_vectors
   A = np.array([a/a.unit, b/a.unit, c/a.unit])
   # Compute volume of parallelepiped.
   volume = np.linalg.det(A) * a.unit**3
   return volume

def compute_mass(system):
   """ Compute the total mass of an OpenMM system. """
   mass = 0.0 * units.amu
   for i in range(system.getNumParticles()):
      mass += system.getParticleMass(i)
   return mass

def run_simulation(pdb):
   """ Run a NPT simulation and gather statistics. """

   # Create the test system.
   forcefield = ForceField(sys.argv[2])
   system = forcefield.createSystem(pdb.topology, **direct_kwargs)
   barostat = openmm.MonteCarloBarostat(pressure, temperature, barostat_frequency)
   # Add barostat.
   system.addForce(barostat)
   # Create integrator.
   integrator = openmm.LangevinIntegrator(temperature, collision_frequency, timestep)        
   # Create simulation class.
   simulation = Simulation(pdb.topology, system, integrator)
   # Set initial positions.
   simulation.context.setPositions(pdb.positions)
   # Assign velocities.
   velocities = generateMaxwellBoltzmannVelocities(system, temperature)
   simulation.context.setVelocities(velocities)
   # Print out the platform used by the context
   if verbose: print "I'm using the platform", simulation.context.getPlatform().getName()
   # Serialize the system if we want.
   Serialize = 0
   if Serialize:
      serial = openmm.XmlSerializer.serializeSystem(system)
      with open('system.xml','w') as f: f.write(serial)
   #==========================================#
   # Computing a bunch of initial values here #
   #==========================================#
   # Show initial system volume.
   box_vectors = system.getDefaultPeriodicBoxVectors()
   volume = compute_volume(box_vectors)
   if verbose: print "initial system volume = %.1f nm^3" % (volume / units.nanometers**3)
   # Determine number of degrees of freedom.
   kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
   ndof = 3*system.getNumParticles() - system.getNumConstraints()
   # Compute total mass.
   mass = compute_mass(system).in_units_of(units.gram / units.mole) /  units.AVOGADRO_CONSTANT_NA # total system mass in g
   if verbose: print "The total mass of the system is", mass / 216 * units.AVOGADRO_CONSTANT_NA
   # Initialize statistics.
   data = dict()
   data['time'] = units.Quantity(np.zeros([niterations], np.float64), units.picoseconds)
   data['potential'] = units.Quantity(np.zeros([niterations], np.float64), units.kilocalories_per_mole)
   data['kinetic'] = units.Quantity(np.zeros([niterations], np.float64), units.kilocalories_per_mole)
   data['volume'] = units.Quantity(np.zeros([niterations], np.float64), units.angstroms**3)
   data['density'] = units.Quantity(np.zeros([niterations], np.float64), units.gram / units.centimeters**3)
   data['kinetic_temperature'] = units.Quantity(np.zeros([niterations], np.float64), units.kelvin)
   # More data structures; stored coordinates, box sizes, densities, and potential energies
   xyzs = []
   boxes = []
   rhos = []
   energies = []
   #========================#
   # Now run the simulation #
   #========================#
   # Equilibrate.
   if verbose: print "Equilibrating..."
   for iteration in range(nequiliterations):
      simulation.step(nsteps)
      state = simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=True,getForces=True)
      kinetic = state.getKineticEnergy()
      potential = state.getPotentialEnergy()
      box_vectors = state.getPeriodicBoxVectors()
      volume = compute_volume(box_vectors)
      density = (mass / volume).in_units_of(units.gram / units.centimeter**3)
      kinetic_temperature = 2.0 * kinetic / kB / ndof # (1/2) ndof * kB * T = KE
      if verbose and (iteration%nprint==0):
         print "%6d %9.3f %16.3f %16.3f %16.4f %10.6f" % (iteration, state.getTime() / units.picoseconds, 
                                                          kinetic_temperature / units.kelvin, potential / units.kilocalories_per_mole, 
                                                          volume / units.nanometers**3, density / (units.gram / units.centimeter**3))
   # Collect production data.
   if verbose: print "Production..."
   #simulation.reporters.append(DCDReporter('dynamics.dcd', 100))
   for iteration in range(niterations):
      # Propagate dynamics.
      simulation.step(nsteps)
      # Compute properties.
      state = simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=True,getForces=True)
      kinetic = state.getKineticEnergy()
      potential = state.getPotentialEnergy()
      box_vectors = state.getPeriodicBoxVectors()
      volume = compute_volume(box_vectors)
      density = (mass / volume).in_units_of(units.gram / units.centimeter**3)
      kinetic_temperature = 2.0 * kinetic / kB / ndof
      if verbose and (iteration%nprint==0):
         print "%6d %9.3f %16.3f %16.3f %16.3f %10.6f" % (iteration, state.getTime() / units.picoseconds, kinetic_temperature / units.kelvin, potential / units.kilocalories_per_mole, volume / units.nanometers**3, density / (units.gram / units.centimeter**3))
      # Store properties.
      data['time'][iteration] = state.getTime() 
      data['potential'][iteration] = potential 
      data['kinetic'][iteration] = kinetic
      data['volume'][iteration] = volume
      data['density'][iteration] = density
      data['kinetic_temperature'][iteration] = kinetic_temperature
      xyzs.append(state.getPositions())
      boxes.append(state.getPeriodicBoxVectors())
      rhos.append(density * 1000)
      energies.append(potential) / units.kilojoules_per_mole
   return data, xyzs, boxes, rhos
   
def analyze(data):
   """Analyze the data from the run_simulation function."""

   #===========================================================================================#
   # Compute statistical inefficiencies to determine effective number of uncorrelated samples. #
   #===========================================================================================#
   data['g_potential'] = statisticalInefficiency(data['potential'] / units.kilocalories_per_mole)
   data['g_kinetic'] = statisticalInefficiency(data['kinetic'] / units.kilocalories_per_mole, fast=True)
   data['g_volume'] = statisticalInefficiency(data['volume'] / units.angstroms**3, fast=True)
   data['g_density'] = statisticalInefficiency(data['density'] / (units.gram / units.centimeter**3), fast=True)
   data['g_kinetic_temperature'] = statisticalInefficiency(data['kinetic_temperature'] / units.kelvin, fast=True)
   
   #=========================================#
   # Compute expectations and uncertainties. #
   #=========================================#
   statistics = dict()
   # Kinetic energy.
   statistics['KE']  = (data['kinetic'] / units.kilocalories_per_mole).mean() * units.kilocalories_per_mole
   statistics['dKE'] = (data['kinetic'] / units.kilocalories_per_mole).std() / np.sqrt(niterations / data['g_kinetic']) * units.kilocalories_per_mole
   statistics['g_KE'] = data['g_kinetic'] * nsteps * timestep 
   # Density
   unit = (units.gram / units.centimeter**3)
   statistics['density']  = (data['density'] / unit).mean() * unit
   statistics['ddensity'] = (data['density'] / unit).std() / np.sqrt(niterations / data['g_density']) * unit
   statistics['g_density'] = data['g_density'] * nsteps * timestep
   # Volume
   unit = units.nanometer**3
   statistics['volume']  = (data['volume'] / unit).mean() * unit
   statistics['dvolume'] = (data['volume'] / unit).std() / np.sqrt(niterations / data['g_volume']) * unit
   statistics['g_volume'] = data['g_volume'] * nsteps * timestep
   statistics['std_volume']  = (data['volume'] / unit).std() * unit
   statistics['dstd_volume'] = (data['volume'] / unit).std() / np.sqrt((niterations / data['g_volume'] - 1) * 2.0) * unit # uncertainty expression from Ref [1].
   # Kinetic temperature
   unit = units.kelvin
   statistics['kinetic_temperature']  = (data['kinetic_temperature'] / unit).mean() * unit
   statistics['dkinetic_temperature'] = (data['kinetic_temperature'] / unit).std() / np.sqrt(niterations / data['g_kinetic_temperature']) * unit
   statistics['g_kinetic_temperature'] = data['g_kinetic_temperature'] * nsteps * timestep

   #==========================#
   # Print summary statistics #
   #==========================#
   print "Summary statistics (%.3f ns equil, %.3f ns production)" % (nequiliterations * nsteps * timestep / units.nanoseconds, niterations * nsteps * timestep / units.nanoseconds)
   print
   # Kinetic energies
   print "Average kinetic energy: %11.6f +- %11.6f  kcal/mol  (g = %11.6f ps)" % (statistics['KE'] / units.kilocalories_per_mole, statistics['dKE'] / units.kilocalories_per_mole, statistics['g_KE'] / units.picoseconds)
   # Kinetic temperature
   unit = units.kelvin
   print "Average kinetic temperature: %11.6f +- %11.6f  K         (g = %11.6f ps)" % (statistics['kinetic_temperature'] / unit, statistics['dkinetic_temperature'] / unit, statistics['g_kinetic_temperature'] / units.picoseconds)
   unit = (units.nanometer**3)
   print "Volume: mean %11.6f +- %11.6f  nm^3" % (statistics['volume'] / unit, statistics['dvolume'] / unit),
   print "g = %11.6f ps" % (statistics['g_volume'] / units.picoseconds)
   print
   unit = (units.gram / units.centimeter**3)
   print "Density: mean %11.6f +- %11.6f  nm^3" % (statistics['density'] / unit, statistics['ddensity'] / unit),
   print "g = %11.6f ps" % (statistics['g_density'] / units.picoseconds)

def energy_driver(mvals,pdb,FF,xyzs):

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
   @return G First derivative of the energies in a N_param x N_coord array
   @return Hd Second derivative of the energies (i.e. diagonal Hessian elements) in a N_param x N_coord array

   """

   # Print the force field XML from the ForceBalance object, with modified parameters.
   pvals = FF.make(os.getcwd(),mvals,False)
   # Load the force field XML file to make the OpenMM object.
   forcefield = ForceField(sys.argv[2])
   # Create the system, setup the simulation.
   system = forcefield.createSystem(pdb.topology, **direct_kwargs)
   integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
   simulation = Simulation(pdb.topology, system, integrator)
   E = []
   # Loop through the snapshots
   for xyz in xyzs:
      # Set the positions using the trajectory
      simulation.context.setPositions(xyz)
      # Compute the potential energy and append to list
      Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
      E.append(Energy)
   return np.array(E)

def energy_derivatives(mvals,pdb,FF,xyzs):

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
   @return G First derivative of the energies in a N_param x N_coord array
   @return Hd Second derivative of the energies (i.e. diagonal Hessian elements) in a N_param x N_coord array

   """

   G        = np.zeros(FF.np,len(xyzs))
   Hd       = np.zeros(FF.np,len(xyzs))
   for i in range(FF.np):
      G[i,:], Hd[i,:] = f12d3p(fdwrap(energy_driver,mvals,i,key=None,pdb=pdb,FF=FF,xyzs=xyzs),FF.h)
   return G, Hd
       
def main():
   
   """ 
   Usage: (runcuda.sh) supernpt.py protein.pdb forcefield.xml <temperature> <pressure>

   This program is meant to be called automatically by ForceBalance
   (specifically, subroutines in openmmio.py).  It is not easy to use
   manually.  This is because the force field is read in from a
   ForceBalance 'FF' class.

   I wrote this program because automatic fitting of the density (or
   other equilibrium properties) is computationally intensive, and the
   calculations need to be distributed to the queue.  The main instance
   of ForceBalance (running on my workstation) queues up a bunch of these
   jobs (using Work Queue).  Then, I submit a bunch of workers to GPU
   clusters (e.g. Certainty, Keeneland).  The worker scripts connect to
   the main instance and receives one of these jobs.

   Of course this script can also be executed locally.  Just make sure
   you have the pickled 'forcebalance.p' file.

   """
   
   # Select platform.
   platform = openmm.Platform.getPlatformByName('Cuda')
   # Set the CUDA device to the environment variable or zero otherwise
   cuda_device = os.environ.get('CUDA_DEVICE',"0")
   print "Setting CUDA Device to", cuda_device
   platform.setPropertyDefaultValue("CudaDevice", cuda_device)
   # Specify the PDB here so we may make the Simulation class.
   pdb = PDBFile(sys.argv[1])
   # Load the force field in from the ForceBalance pickle.
   FF,mvals = lp_load(open('forcebalance.p'))
   # Create the force field XML files.
   pvals = FF.make(os.getcwd(),mvals,False)
   # Run the simulation.
   Data, Xyzs, Boxes, Rhos = run_simulation(pdb)
   # Get statistics from our simulation.
   analyze(Data)
   # Now that we have the coordinates, we can compute the energy derivatives.
   G, Hd = energy_derivatives(mvals, pdb, FF, Xyzs)
   # The density derivative can be computed using the energy derivative.
   N = len(Xyzs)
   mkT = -1 * temperature * boltzmann_constant
   # The density derivative.
   GRho = mKT * ((np.mat(G) * col(Rhos)) / N - np.mean(Rhos) * np.mean(G, axis=1))
   print GRho

if __name__ == "__main__":
   main()
