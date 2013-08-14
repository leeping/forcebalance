#!/usr/bin/env python

"""
@package npt

Runs a simulation to compute condensed phase properties (for example, the density 
or the enthalpy of vaporization) and compute the derivative with respect 
to changing the force field parameters.  This script is a part of ForceBalance.

The basic idea is this: First we run a density simulation to determine
the average density.  This quantity of course has some uncertainty,
and in general we want to avoid evaluating finite-difference
derivatives of noisy quantities.  The key is to realize that the
densities are sampled from a Boltzmann distribution, so the analytic
derivative can be computed if the potential energy derivative is
accessible.  We compute the potential energy derivative using
finite-difference of snapshot energies and apply a simple formula to
compute the density derivative.

References

[1] Shirts MR, Mobley DL, Chodera JD, and Pande VS. Accurate and efficient corrections for
missing dispersion interactions in molecular simulations. JPC B 111:13052, 2007.

[2] Ahn S and Fessler JA. Standard errors of mean, variance, and standard deviation estimators.
Technical Report, EECS Department, The University of Michigan, 2003.

Copyright And License

@author Lee-Ping Wang <leeping@stanford.edu>
@author John D. Chodera <jchodera@gmail.com> (Wrote statisticalInefficiency and MTS-VVVR)

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

#==================#
#| Global Imports |#
#==================#

import os
import sys
import glob
import shutil
import argparse
import traceback
import numpy as np
from copy import deepcopy
from collections import namedtuple
from forcebalance.forcefield import FF
from forcebalance.nifty import col, flat, lp_dump, lp_load, printcool, printcool_dictionary, statisticalInefficiency, which, _exec, isint
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, f1d7p, in_fd
from forcebalance.molecule import Molecule

#========================================================#
#| Global, user-tunable variables (simulation settings) |#
#========================================================#

# TODO: Strip out PDB and XML file arguments.
# TODO: Strip out all units.

parser = argparse.ArgumentParser()
parser.add_argument('engine', help='MD program that we are using; choose "openmm" or "gromacs"')
parser.add_argument('liquid_prod_steps', type=int, help='Number of time steps for the liquid production simulation')
parser.add_argument('liquid_timestep', type=float, help='Length of the time step for the liquid simulation, in femtoseconds')
parser.add_argument('liquid_interval', type=float, help='Time interval for saving the liquid coordinates, in picoseconds')
parser.add_argument('temperature',type=float, help='Temperature (K)')
parser.add_argument('pressure',type=float, help='Pressure (Atm)')

# Other optional arguments
parser.add_argument('--liquid_equ_steps', type=int, help='Number of time steps used for equilibration', default=100000)
parser.add_argument('--gas_equ_steps', type=int, help='Number of time steps for the gas-phase equilibration simulation', default=100000)
parser.add_argument('--gas_prod_steps', type=int, help='Number of time steps for the gas-phase production simulation', default=1000000)
parser.add_argument('--gas_timestep', type=float, help='Time step for the gas-phase simulation, in femtoseconds', default=0.5)
parser.add_argument('--gas_interval', type=float, help='Time interval for saving the gas-phase coordinates, in picoseconds', default=0.1)
parser.add_argument('--anisotropic', action='store_true', help='Enable anisotropic scaling of periodic box (useful for crystals)')
parser.add_argument('--mts_vvvr', action='store_true', help='Enable multiple timestep integrator (OpenMM)')
parser.add_argument('--nt', type=int, help='Number of threads when executing GROMACS', default=1)
parser.add_argument('--force_cuda', action='store_true', help='Exit if CUDA platform is not available (OpenMM)')
parser.add_argument('--gmxpath', type=str, help='Specify the location of GROMACS executables', default="")
parser.add_argument('--minimize_energy', action='store_true', help='Minimize the energy of the system prior to running dynamics')

args = parser.parse_args()

# Simulation settings for the condensed phase system.
timestep         = float(args.liquid_timestep)                                 # timestep for integration in femtosecond
faststep         = 0.25                                                        # "fast" timestep (for MTS integrator, if used)
nsteps           = int(1000 * args.liquid_interval / args.liquid_timestep)     # Interval for saving snapshots (in steps)
nequil           = args.liquid_equ_steps / nsteps                              # Number of snapshots set aside for equilibration
nprod            = args.liquid_prod_steps / nsteps                             # Number of snapshots in production run
temperature      = args.temperature                                            # temperature in kelvin
pressure         = args.pressure                                               # pressure in atmospheres

# Simulation settings for the monomer.
m_timestep          = float(args.gas_timestep)
m_nsteps            = int(1000 * args.gas_interval / args.gas_timestep)
m_nequil            = args.gas_equ_steps / m_nsteps
m_nprod             = args.gas_prod_steps / m_nsteps

if args.engine == "openmm":
    try:
        from simtk.unit import *
        from simtk.openmm import *
        from simtk.openmm.app import *
        from forcebalance.openmmio import *
    except:
        traceback.print_exc()
        raise Exception("Cannot import OpenMM modules")
elif args.engine == "gromacs" or args.engine == "gmx":
    from forcebalance.gmxio import *
    if args.mts_vvvr:
        raise Exception("Selected multiple timestep integrator with GROMACS interface, but it is only usable with OpenMM interface")
    if args.force_cuda:
        raise Exception("Selected CUDA platform with GROMACS interface, but it is only usable with OpenMM interface")
else:
    raise Exception('Only OpenMM and GROMACS support implemented at this time.')

printcool("ForceBalance condensed phase simulation using engine: %s" % args.engine)
print "For the condensed phase system, I will collect %i snapshots spaced apart by %i x %.3f fs time steps" % (nprod, nsteps, args.liquid_timestep)
print "For the gas phase system, I will collect %i snapshots spaced apart by %i x %.3f fs time steps" % (m_nprod, m_nsteps, args.gas_timestep)
if nprod < 2:
    raise Exception('Please set the number of liquid time steps so that you collect at least two snapshots (minimum %i)' % (2 * nsteps))
if m_nprod < 2:
    raise Exception('Please set the number of gas time steps so that you collect at least two snapshots (minimum %i)' % (2 * m_nsteps))

DoEDA = True
if args.mts_vvvr:
    # EDA relies on Force Groups, as does MTS integrator, so they are incompatible.
    DoEDA = False

#==================#
#|   Subroutines  |#
#==================#

def mean_stderr(ts):
    """ Get mean and standard deviation of a time series. """
    return np.mean(ts), np.std(ts)*np.sqrt(statisticalInefficiency(ts, warn=False)/len(ts))

def bzavg(obs,boltz):
    """ Get the Boltzmann average of an observable. """
    if obs.ndim == 2:
        if obs.shape[0] == len(boltz) and obs.shape[1] == len(boltz):
            raise Exception('Error - both dimensions have length equal to number of snapshots, now confused!')
        elif obs.shape[0] == len(boltz):
            return np.sum(obs*boltz.reshape(-1,1),axis=0)/np.sum(boltz)
        elif obs.shape[1] == len(boltz):
            return np.sum(obs*boltz,axis=1)/np.sum(boltz)
        else:
            raise Exception('The dimensions are wrong!')
    elif obs.ndim == 1:
        return np.dot(obs,boltz)/sum(boltz)
    else:
        raise Exception('The number of dimensions can only be 1 or 2!')

def PrintEDA(EDA, N):
    # Get energy decomposition statistics.
    PrintDict = OrderedDict()
    for key, val in EDA.items():
        val_avg, val_err = mean_stderr(val)
        if val_avg == 0.0: continue
        if val_err == 0.0: continue
        PrintDict[key] = "% 12.4f +- %10.4f [ % 9.4f +- %7.4f ]" % (val_avg, val_err, val_avg/N, val_err/N)
    printcool_dictionary(PrintDict, "Energy Decomposition Analysis, Mean +- Stderr [Per Molecule] (kJ/mol)")

#=============================================#
#|   Driver classes for simulation software  |#
#=============================================#

class MDEngine(object):
    """
    This class represents the molecular dynamics engine, which controls the execution of a MD code
    (e.g. GROMACS, OpenMM, TINKER) in order to obtain simulation trajectories and potential energy
    derivatives.  This information is used to calculate the condensed phase properties and their
    derivatives in parameter space.
    """

    def __init__(self, FF):
        self.FF = FF

    def energy_derivatives(self,mvals,h,phase,length,AGrad=True,dipole=False):

        """
        Compute the first and second derivatives of a set of snapshot
        energies with respect to the force field parameters.
    
        This basically calls the finite difference subroutine on the
        energy_driver subroutine also in this script.
    
        @param[in] mvals Mathematical parameter values
        @param[in] h Finite difference step size
        @param[in] phase The phase (liquid, gas) to perform the calculation on
        @param[in] AGrad Switch to turn derivatives on or off; if off, return all zeros
        @param[in] dipole Switch for dipole derivatives.
        @return G First derivative of the energies in a N_param x N_coord array
        @return GDx First derivative of the box dipole moment x-component in a N_param x N_coord array
        @return GDy First derivative of the box dipole moment y-component in a N_param x N_coord array
        @return GDz First derivative of the box dipole moment z-component in a N_param x N_coord array
    
        """
        G        = np.zeros((self.FF.np,length))
        GDx      = np.zeros((self.FF.np,length))
        GDy      = np.zeros((self.FF.np,length))
        GDz      = np.zeros((self.FF.np,length))
        if not AGrad:
            return G, GDx, GDy, GDz
        ED0      = self.energy_driver(mvals, phase, dipole=dipole)
        for i in range(self.FF.np):
            print self.FF.plist[i] + " "*30
            EDG, _   = f12d3p(fdwrap(self.energy_driver,mvals,i,key=None,phase=phase,dipole=dipole,resetvs='VirtualSite' in self.FF.plist[i]),h,f0=ED0)
            if dipole:
                G[i,:]   = EDG[:,0]
                GDx[i,:] = EDG[:,1]
                GDy[i,:] = EDG[:,2]
                GDz[i,:] = EDG[:,3]
            else:
                G[i,:]   = EDG[:]
        return G, GDx, GDy, GDz

    def property_derivatives(self,mvals,h,phase,kT,property_driver,property_kwargs,AGrad=True):

        """ 
        Function for double-checking property derivatives.  This function is called to perform
        a more explicit numerical derivative of the property, rather than going through the 
        fluctuation formula.  It takes longer and is potentially less precise, which means
        it's here mainly as a sanity check.

        @param[in] mvals Mathematical parameter values
        @param[in] h Finite difference step size
        @param[in] phase The phase (liquid, gas) to perform the calculation on
        @param[in] property_driver The function that calculates the property
        @param[in] property_driver A dictionary of arguments that goes into calculating the property
        @param[in] AGrad Switch to turn derivatives on or off; if off, return all zeros
        @return G First derivative of the property

        """
        G        = np.zeros(self.FF.np)
        if not AGrad:
            return G
        ED0      = self.energy_driver(mvals, phase, dipole=True)
        E0       = ED0[:,0]
        D0       = ED0[:,1:]
        P0       = property_driver(None, **property_kwargs)
        if 'h_' in property_kwargs:
            H0 = property_kwargs['h_'].copy()
        for i in range(self.FF.np):
            print self.FF.plist[i] + " "*30
            ED1 = fdwrap(self.energy_driver,mvals,i,key=None,phase=phase,dipole=True,resetvs='VirtualSite' in self.FF.plist[i])(h)
            E1       = ED1[:,0]
            D1       = ED1[:,1:]
            b = np.exp(-(E1-E0)/kT)
            b /= np.sum(b)
            if 'h_' in property_kwargs:
                property_kwargs['h_'] = H0.copy() + (E1-E0)
            if 'd_' in property_kwargs:
                property_kwargs['d_'] = D1.copy()
            S = -1*np.dot(b,np.log(b))
            InfoContent = np.exp(S)
            if InfoContent / len(E0) < 0.1:
                print "Warning: Effective number of snapshots: % .1f (out of %i)" % (InfoContent, len(E0))
            P1 = property_driver(b=b,**property_kwargs)
            EDM1 = fdwrap(self.energy_driver,mvals,i,key=None,phase=phase,dipole=True,resetvs='VirtualSite' in self.FF.plist[i])(-h)
            EM1       = EDM1[:,0]
            DM1       = EDM1[:,1:]
            b = np.exp(-(EM1-E0)/kT)
            b /= np.sum(b)
            if 'h_' in property_kwargs:
                property_kwargs['h_'] = H0.copy() + (EM1-E0)
            if 'd_' in property_kwargs:
                property_kwargs['d_'] = DM1.copy()
            S = -1*np.dot(b,np.log(b))
            InfoContent = np.exp(S)
            if InfoContent / len(E0) < 0.1:
                print "Warning: Effective number of snapshots: % .1f (out of %i)" % (InfoContent, len(E0))
            PM1 = property_driver(b=b,**property_kwargs)
            G[i] = (P1-PM1)/(2*h)
        if 'h_' in property_kwargs:
            property_kwargs['h_'] = H0.copy()
        if 'd_' in property_kwargs:
            property_kwargs['d_'] = D0.copy()
        return G

class Gromacs_MD(MDEngine):
    def __init__(self, FF):
        super(Gromacs_MD,self).__init__(FF)
        if args.gmxpath != '' and os.path.exists(os.path.join(args.gmxpath,'grompp')):
            self.gmxpath = args.gmxpath
        else:
            if which('grompp') != '':
                self.gmxpath = which('grompp')
            else:
                raise RuntimeError("Failed to discern location of Gromacs executables, make sure it's in your PATH")
        # Execute "grompp -h" to ensure that GROMACS can run properly.
        _exec(os.path.join(self.gmxpath,'grompp -h'), print_to_screen=True)
        # Global options for liquid and gas
        self.opts = {"liquid" : {"comm_mode" : "linear"},
                     "gas" : {"comm_mode" : "None", "nstcomm" : 0}}

    def callgmx(self, command, stdin=None, print_to_screen=True, print_command=True, **kwargs):
        # Call a GROMACS program as you would from the command line.
        csplit = command.split()
        prog = os.path.join(self.gmxpath, csplit[0])
        csplit[0] = prog
        return _exec(' '.join(csplit), stdin=stdin, print_to_screen=print_to_screen, print_command=print_command, **kwargs)
   
    def run_simulation(self, phase, minimize=True, savexyz=True):
        # Edit the mdp file (trajectory save frequency, number of time steps).
        # Minimize the energy if necessary.
        # Remember to save the trajectory.
        # Extract the quantities.
        # If running remotely, make sure GROMACS is in your PATH!
        # Arguments for running equilibration.
        eq_opts = {"liquid" : dict({"integrator" : "md", "nsteps" : args.liquid_equ_steps,
                                    "dt" : args.liquid_timestep / 1000,
                                    "ref_t" : args.temperature, "gen_temp" : args.temperature,
                                    "pcoupl" : "berendsen", "ref_p" : args.pressure}, **self.opts["liquid"]),
                   "gas" : dict({"integrator" : "md", "nsteps" : args.gas_equ_steps,
                                 "dt" : args.gas_timestep / 1000,
                                 "ref_t" : args.temperature, "gen_temp" : args.temperature}, **self.opts["gas"])}
        # Arguments for running production.
        md_opts = {"liquid" : dict({"integrator" : "md", "nsteps" : args.liquid_prod_steps,
                                    "dt" : args.liquid_timestep / 1000,
                                    "ref_t" : args.temperature, "gen_temp" : args.temperature,
                                    "pcoupl" : "parrinello-rahman", "ref_p" : args.pressure}, **self.opts["liquid"]),
                   "gas" : dict({"integrator" : "md", "nsteps" : args.gas_prod_steps,
                                 "dt" : args.gas_timestep / 1000,
                                 "ref_t" : args.temperature, "gen_temp" : args.temperature}, **self.opts["gas"])}
        # Arguments for running minimization.
        min_opts = {"liquid" : dict({"integrator" : "steep", "emtol" : 10.0, "nsteps" : 10000}, **self.opts["liquid"]),
                    "gas" : dict({"integrator" : "steep", "emtol" : 10.0, "nsteps" : 10000}, **self.opts["gas"])}
        # Arguments for not saving coordinates.
        nosave_opts = {"nstxout" : 0, "nstenergy" : 0}
        # Arguments for saving coordinates.
        save_opts = {"nstxout" : nsteps, "nstenergy" : nsteps, "nstcalcenergy" : nsteps}
        # Minimize the energy.
        if minimize:
            edit_mdp("%s.mdp" % phase, "%s-min.mdp" % phase, dict(min_opts[phase], **save_opts), verbose=True)
            self.callgmx("grompp -maxwarn 1 -c %s.gro -p %s.top -f %s-min.mdp -o %s-min.tpr" % (phase, phase, phase, phase))
            self.callgmx("mdrun -v -deffnm %s-min" % phase)
        # Run equilibration.
        edit_mdp("%s.mdp" % phase, "%s-eq.mdp" % phase, dict(eq_opts[phase], **save_opts), verbose=True)
        self.callgmx("grompp -maxwarn 1 -c %s-min.gro -p %s.top -f %s-eq.mdp -o %s-eq.tpr" % (phase, phase, phase, phase))
        self.callgmx("mdrun -v -deffnm %s-eq -nt %i -stepout %i" % (phase, args.nt, nsteps), expand_cr=True)
        if int(eq_opts[phase]["nsteps"]) == 0:
            shutil.copy2("%s-min.gro" % phase, "%s-eq.gro" % phase)
        # Run production.
        edit_mdp("%s.mdp" % phase, "%s-md.mdp" % phase, dict(md_opts[phase], **save_opts), verbose=True)
        self.callgmx("grompp -maxwarn 1 -c %s-eq.gro -p %s.top -f %s-md.mdp -o %s-md.tpr" % (phase, phase, phase, phase))
        self.callgmx("mdrun -v -deffnm %s-md -nt %i -stepout %i" % (phase, args.nt, nsteps), outfnm="%s-md.out" % phase, copy_stderr=True, expand_cr=True)
        # After production, run analysis.
        self.callgmx("g_dipoles -s %s-md.tpr -f %s-md.trr -o %s-md-dip.xvg -xvg no" % (phase, phase, phase), stdin="System\n")
        # Figure out which energy terms need to be printed.
        o = self.callgmx("g_energy -f %s-md.edr -xvg no" % (phase), stdin="Total-Energy\n", copy_stderr=True)
        parsemode = 0
        energyterms = OrderedDict()
        for line in o:
            s = line.split()
            if "Select the terms you want from the following list" in line:
                parsemode = 1
            if parsemode == 1:
                if len(s) > 0 and all([isint(i) for i in s[::2]]):
                    parsemode = 2
            if parsemode == 2:
                # g_energy garbles stdout and stderr so our parser
                # needs to be more robust.
                if len(s) > 0:
                    try:
                        if all([isint(i) for i in s[::2]]):
                            for j in range(len(s))[::2]:
                                num = int(s[j])
                                name = s[j+1]
                                energyterms[name] = num
                    except: pass
                # else:
                #     parsemode = 0
        ekeep = [k for k,v in energyterms.items() if v <= energyterms['Total-Energy']]
        ekeep += ['Volume', 'Density']
        o = self.callgmx("g_energy -f %s-md.edr -o %s-md-energy.xvg -xvg no" % (phase, phase), stdin="\n".join(ekeep))
        # for line in o:
        #     print line
        edecomp = OrderedDict()
        Rhos = []
        Volumes = []
        Kinetics = []
        Potentials = []
        for line in open("%s-md-energy.xvg" % phase):
            s = [float(i) for i in line.split()]
            for i in range(len(ekeep) - 2):
                val = s[i+1]
                if ekeep[i] in edecomp:
                    edecomp[ekeep[i]].append(val)
                else:
                    edecomp[ekeep[i]] = [val]
            Rhos.append(s[-1])
            Volumes.append(s[-2])
        Rhos = np.array(Rhos)
        Volumes = np.array(Volumes)
        Potentials = np.array(edecomp['Potential'])
        Kinetics = np.array(edecomp['Kinetic-En.'])
        Dips = np.array([[float(i) for i in line.split()[1:4]] for line in open("%s-md-dip.xvg" % phase)])
        for i in glob.glob("#*"):
            os.remove(i)
        return Rhos, Potentials, Kinetics, Volumes, Dips, OrderedDict([(key, np.array(val)) for key, val in edecomp.items()])
    
    def energy_driver(self,mvals,phase,verbose=False,dipole=False,resetvs=False):
        # Print the force field XML from the ForceBalance object, with modified parameters.
        self.FF.make(mvals)
        # Arguments for running over snapshots.
        shot_opts = {"liquid" : dict({"integrator" : "md", "nsteps" : 0, "nstenergy" : 1}, **self.opts["liquid"]),
                     "gas" : dict({"integrator" : "md", "nsteps" : 0, "nstenergy" : 1}, **self.opts["gas"])}
        # Run over the snapshots.
        if in_fd(): verbose=False
        edit_mdp("%s.mdp" % phase, "%s-shot.mdp" % phase, shot_opts[phase], verbose=verbose)
        self.callgmx("grompp -maxwarn 1 -c %s.gro -p %s.top -f %s-shot.mdp -o %s-shot.tpr" % (phase, phase, phase, phase), print_command=verbose, print_to_screen=verbose)
        self.callgmx("mdrun -v -deffnm %s-shot -rerun %s-md.trr -rerunvsite -nt %i" % (phase, phase, args.nt), print_command=verbose, print_to_screen=verbose)
        # Get potential energies.
        self.callgmx("g_energy -f %s-shot.edr -o %s-shot-energy.xvg -xvg no" % (phase, phase), stdin="Potential\n", print_command=verbose, print_to_screen=verbose)
        E = [float(line.split()[1]) for line in open("%s-shot-energy.xvg" % phase)]
        print "\r",
        if verbose: print E
        if dipole:
            # Return a Nx4 array with energies in the first column and dipole in columns 2-4.
            self.callgmx("g_dipoles -s %s-shot.tpr -f %s-md.trr -o %s-shot-dip.xvg -xvg no" % (phase, phase, phase), stdin="System\n", print_command=verbose, print_to_screen=verbose)
            D = np.array([[float(i) for i in line.split()[1:4]] for line in open("%s-shot-dip.xvg" % phase)])
            for i in glob.glob("#*"): os.remove(i)
            return np.hstack((np.array(E).reshape(-1,1), np.array(D).reshape(-1,3)))
        else:
            # Return just the energies.
            for i in glob.glob("#*"): os.remove(i)
            return np.array(E)
    
class OpenMM_MD(MDEngine):
    def __init__(self, FF):
        super(OpenMM_MD,self).__init__(FF)
        # Langevin integrator friction / random force parameter in OpenMM, in inverse picoseconds
        self.collision_parameter  = 1.0
        # number of steps between MC volume adjustments
        self.barostat_interval   = 25
        # Keyword arguments (settings) for setting up the system.
        # AMOEBA mutual
        mutual_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*nanometer,
                         'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 0.85,
                         'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                         'mutualInducedTargetEpsilon' : 1e-6, 'useDispersionCorrection' : True}
        # AMOEBA direct
        direct_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*nanometer,
                         'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 0.85,
                         'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                         'polarization' : 'direct', 'useDispersionCorrection' : True}
        # AMOEBA nonpolarizable
        nonpol_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.7*nanometer,
                         'constraints' : None, 'rigidWater' : False, 'vdwCutoff' : 0.85,
                         'aEwald' : 5.4459052, 'pmeGridDimensions' : [24,24,24],
                         'useDispersionCorrection' : True}
        # TIP3P and other rigid water models
        tip3p_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.85*nanometer,
                        'useSwitchingFunction' : True, 'switchingDistance' : 0.75*nanometer,
                        'constraints' : HBonds, 'rigidwater' : True, 'useDispersionCorrection' : True}
        # General periodic systems (no constraints)
        gen_kwargs = {'nonbondedMethod' : PME, 'nonbondedCutoff' : 0.85*nanometer,
                      'useSwitchingFunction' : True, 'switchingDistance' : 0.75*nanometer,
                      'constraints' : None, 'rigidwater' : False, 'useDispersionCorrection' : True}
        # Same settings for the monomer.
        m_mutual_kwargs = {'nonbondedMethod' : NoCutoff, 'constraints' : None,
                           'rigidWater' : False, 'mutualInducedTargetEpsilon' : 1e-6, 'removeCMMotion' : False}
        m_direct_kwargs = {'nonbondedMethod' : NoCutoff, 'constraints' : None,
                           'rigidWater' : False, 'polarization' : 'direct', 'removeCMMotion' : False}
        m_nonpol_kwargs = {'nonbondedMethod' : NoCutoff, 'constraints' : None,
                           'rigidWater' : False, 'removeCMMotion' : False}
        m_tip3p_kwargs = {'nonbondedMethod' : NoCutoff, 'rigidwater' : True, 'removeCMMotion' : False}
        m_gen_kwargs = {'nonbondedMethod' : NoCutoff, 'rigidwater' : False, 'removeCMMotion' : False}
        # Dictionary of simulation, system, and trajectory objects.
        self.Simulations = OrderedDict()
        self.Systems = OrderedDict()
        self.Xyzs = OrderedDict()
        self.Boxes = OrderedDict()
        self.PlatNames = {"gas":"Reference",
                          "liquid":"CUDA"}
        # Dictionary of PDB and simulation settings
        self.PDBs = OrderedDict()
        self.Settings = OrderedDict()
        # ForceBalance force field object.
        self.FF = FF
        # This creates a system from a force field XML file.
        forcefield = ForceField(FF.openmmxml)
        self.ffxml = FF.openmmxml
        # Read in the PDB files here.
        if os.path.exists("gas.pdb"):
            self.PDBs["gas"] = PDBFile("gas.pdb")
        elif os.path.exists("mono.pdb"):
            self.PDBs["gas"] = PDBFile("mono.pdb")
        if os.path.exists("liquid.pdb"):
            self.PDBs["liquid"] = PDBFile("liquid.pdb")
        elif os.path.exists("conf.pdb"):
            self.PDBs["liquid"] = PDBFile("conf.pdb")
        # Try to detect if we're using an AMOEBA system.
        if any(['Amoeba' in i.__class__.__name__ for i in forcefield._forces]):
            print "Detected AMOEBA system!"
            if FF.amoeba_pol == "mutual":
                print "Setting mutual polarization"
                self.Settings["liquid"] = mutual_kwargs
                self.Settings["gas"] = m_mutual_kwargs
            elif FF.amoeba_pol == "direct":
                print "Setting direct polarization"
                self.Settings["liquid"] = direct_kwargs
                self.Settings["gas"] = m_direct_kwargs
            else:
                print "No polarization"
                self.Settings["liquid"] = nonpol_kwargs
                self.Settings["gas"] = m_nonpol_kwargs
        else:
            if 'tip' in self.ffxml:
                print "Using rigid water."
                self.Settings["liquid"] = tip3p_kwargs
                self.Settings["gas"] = m_tip3p_kwargs
            else:
                print "Using general settings for nonpolarizable force fields."
                self.Settings["liquid"] = gen_kwargs
                self.Settings["gas"] = m_gen_kwargs
    
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
    
    def create_simulation_object(self, phase, precision="mixed"):
        """ Create an OpenMM simulation object. """
        pbc = phase == "liquid"
        platname = self.PlatNames[phase]
        # Create the platform.
        if platname == "CUDA":
            try:
                platform = Platform.getPlatformByName(platname)
                # Set the device to the environment variable or zero otherwise
                device = os.environ.get('CUDA_DEVICE',"0")
                print "Setting Device to", device
                platform.setPropertyDefaultValue("CudaDeviceIndex", device)
                platform.setPropertyDefaultValue("CudaPrecision", precision)
                platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
                cpupme = os.environ.get('CPU_PME',"n")
                if cpupme.lower() == "y":
                    platform.setPropertyDefaultValue("CudaUseCpuPme", "true")
            except:
                traceback.print_exc()
                if args.force_cuda:
                    raise Exception('Force CUDA option is enabled but CUDA platform not available')
                platname = "Reference"
        if platname == "Reference":
            platform = Platform.getPlatformByName(platname)
        # Create the system.
        try:
            pdb = self.PDBs[phase]
            settings = self.Settings[phase]
        except:
            traceback.print_exc()
            raise RuntimeError("Tried to load pdb and settings for %s phase but failed" % phase)
        forcefield = ForceField(self.ffxml)
        mod = Modeller(pdb.topology, pdb.positions)
        mod.addExtraParticles(forcefield)
        system = forcefield.createSystem(mod.topology, **settings)
        if pbc:
            if args.anisotropic:
                barostat = MonteCarloAnisotropicBarostat([pressure, pressure, pressure]*atmospheres, temperature*kelvin, self.barostat_interval)
            else:
                barostat = MonteCarloBarostat(pressure*atmospheres, temperature*kelvin, self.barostat_interval)
            system.addForce(barostat)

        # Create integrator.
        if args.mts_vvvr:
            print "Using new multiple-timestep velocity-verlet with velocity randomization (MTS-VVVR) integrator."
            print "Warning: not proven to work in most situations"
            integrator = MTSVVVRIntegrator(temperature*kelvin, self.collision_parameter/picosecond, timestep*femtosecond, system, ninnersteps=int(timestep/faststep))
        else:
            integrator = LangevinIntegrator(temperature*kelvin, self.collision_parameter/picosecond, timestep*femtosecond) 
            for n, i in enumerate(system.getForces()):
                print "Setting Force %s to group %i" % (i.__class__.__name__, n)
                i.setForceGroup(n)

        # Delete any pre-existing simulation and system objects.
        if phase in self.Simulations:
            del self.Simulations[phase]
            del self.Systems[phase]

        # Create simulation object; assign positions and velocities.
        simulation = Simulation(mod.topology, system, integrator, platform)
        # Print out the platform properties.
        print "I'm using the platform", simulation.context.getPlatform().getName()
        printcool_dictionary({i:simulation.context.getPlatform().getPropertyValue(simulation.context,i) for i in simulation.context.getPlatform().getPropertyNames()},title="Platform %s has properties:" % simulation.context.getPlatform().getName())
        simulation.context.setPositions(mod.positions)
        simulation.context.setVelocitiesToTemperature(temperature*kelvin)
        self.Simulations[phase] = simulation
        self.Systems[phase] = system
    
    def EnergyDecomposition(self, Sim):
        # Before using EnergyDecomposition, make sure each Force is set to a different group.
        EnergyTerms = OrderedDict()
        Potential = Sim.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
        Kinetic = Sim.context.getState(getEnergy=True).getKineticEnergy() / kilojoules_per_mole
        if DoEDA:
            for i in range(Sim.system.getNumForces()):
                EnergyTerms[Sim.system.getForce(i).__class__.__name__] = Sim.context.getState(getEnergy=True,groups=2**i).getPotentialEnergy() / kilojoules_per_mole
        EnergyTerms['Potential'] = Potential
        EnergyTerms['Kinetic'] = Kinetic
        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        Kinetic_Temperature = 2.0 * Kinetic * kilojoule_per_mole / kB / self.ndof / kelvin
        EnergyTerms['Kinetic Temperature'] = Kinetic_Temperature
        EnergyTerms['Total Energy'] = Potential+Kinetic
        return EnergyTerms
    
    def run_simulation(self, phase, minimize=True, savexyz=True):
        """ Run a NPT simulation for a given phase and gather statistics. """
        # Set periodic boundary conditions.
        pbc = phase == "liquid"

        # Create simulation and system objects.
        if phase not in self.Simulations:
            self.create_simulation_object(phase, precision="mixed")
        simulation = self.Simulations[phase]
        system = self.Systems[phase]

        # Minimize the energy.
        if minimize:
            print "Minimizing the energy... (starting energy % .3f kJ/mol)" % simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole),
            simulation.minimizeEnergy()
            print "Done (final energy % .3f kJ/mol)" % simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)

        # Serialize the system if we want.
        Serialize = 0
        if Serialize:
            serial = XmlSerializer.serializeSystem(system)
            with open('%s_system.xml' % phase,'w') as f: f.write(serial)

        kB = BOLTZMANN_CONSTANT_kB * AVOGADRO_CONSTANT_NA
        # Determine number of degrees of freedom; the center of mass motion remover is also a constraint.
        self.ndof = 3*(system.getNumParticles() - sum([system.isVirtualSite(i) for i in range(system.getNumParticles())])) - system.getNumConstraints() - 3*pbc
        # Compute total mass.
        mass = self.compute_mass(system).in_units_of(gram / mole) /  AVOGADRO_CONSTANT_NA # total system mass in g
        # Initialize statistics.
        edecomp = OrderedDict()
        # More data structures; stored coordinates, box sizes, densities, and potential energies
        self.Xyzs[phase] = []
        self.Boxes[phase] = []
        rhos = []
        potentials = []
        kinetics = []
        volumes = []
        dipoles = []
        #========================#
        # Now run the simulation #
        #========================#
        # Equilibrate.
        print "Equilibrating..."
        for iteration in range(nequil):
            simulation.step(nsteps)
            state = simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=False,getForces=False)
            kinetic = state.getKineticEnergy()
            potential = state.getPotentialEnergy()
            if pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (mass / volume).in_units_of(kilogram / meter**3)
            else:
                volume = 0.0 * nanometers ** 3
                density = 0.0 * kilogram / meter ** 3
            kinetic_temperature = 2.0 * kinetic / kB / self.ndof # (1/2) ndof * kB * T = KE
            print "%6d %9.3f %9.3f % 13.3f %10.4f %13.4f" % (iteration, state.getTime() / picoseconds,
                                                             kinetic_temperature / kelvin, potential / kilojoules_per_mole,
                                                             volume / nanometers**3, density / (kilogram / meter**3))
        # Collect production data.
        print "Production..."
        if savexyz:
            simulation.reporters.append(DCDReporter('dynamics.dcd', nsteps))
        for iteration in range(nprod):
            # Propagate dynamics.
            simulation.step(nsteps)
            # Compute properties.
            state = simulation.context.getState(getEnergy=True,getPositions=True,getVelocities=False,getForces=False)
            self.Xyzs[phase].append(state.getPositions())
            kinetic = state.getKineticEnergy()
            potential = state.getPotentialEnergy()
            kinetic_temperature = 2.0 * kinetic / kB / self.ndof
            if pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (mass / volume).in_units_of(kilogram / meter**3)
            else:
                box_vectors = None
                volume = 0.0 * nanometers ** 3
                density = 0.0 * kilogram / meter ** 3
            self.Boxes[phase].append(box_vectors)
            # Perform energy decomposition.
            for comp, val in self.EnergyDecomposition(simulation).items():
                if comp in edecomp:
                    edecomp[comp].append(val)
                else:
                    edecomp[comp] = [val]
            print "%6d %9.3f %9.3f % 13.3f %10.4f %13.4f" % (iteration, state.getTime() / picoseconds, kinetic_temperature / kelvin, potential / kilojoules_per_mole, volume / nanometers**3, density / (kilogram / meter**3))
            rhos.append(density.value_in_unit(kilogram / meter**3))
            potentials.append(potential / kilojoules_per_mole)
            kinetics.append(kinetic / kilojoules_per_mole)
            volumes.append(volume / nanometer**3)
            dipoles.append(get_dipole(simulation,positions=self.Xyzs[phase][-1]))
            
        return np.array(rhos), np.array(potentials), np.array(kinetics), np.array(volumes), np.array(dipoles), OrderedDict([(key, np.array(val)) for key, val in edecomp.items()])
    
    def energy_driver(self,mvals,phase,verbose=False,dipole=False,resetvs=False):
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
        # Set periodic boundary conditions.
        pbc = phase == "liquid"

        # Load in the information for the appropriate phase.
        pdb = self.PDBs[phase]
        xyzs = self.Xyzs[phase]
        simulation = self.Simulations[phase]
        settings = self.Settings[phase]
        boxes = self.Boxes[phase]
        
        # Print the force field XML from the ForceBalance object, with modified parameters.
        self.FF.make(mvals)
        forcefield = ForceField(self.ffxml)

        # Create the system and update the simulation parameters.
        mod = Modeller(pdb.topology, pdb.positions)
        mod.addExtraParticles(forcefield)
        system = forcefield.createSystem(mod.topology, **settings)
        UpdateSimulationParameters(system, simulation)
        E = []
        D = []
        q = None
        for i in simulation.system.getForces():
            if i.__class__.__name__ == "NonbondedForce":
                q = np.array([i.getParticleParameters(j)[0]._value for j in range(i.getNumParticles())])
        
        # Loop through the snapshots
        for xyz,box in zip(self.Xyzs[phase],self.Boxes[phase]):
            # First set the positions in the simulation in order to apply the constraints.
            simulation.context.setPositions(xyz)
            if simulation.system.getNumConstraints() > 0:
                simulation.context.applyConstraints(1e-8)
                xyz0 = simulation.context.getState(getPositions=True).getPositions()
            else:
                xyz0 = xyz
            # Then reset the virtual site positions manually.
            xyz1 = ResetVirtualSites(xyz0, system) if resetvs else xyz0
            simulation.context.setPositions(xyz1)
            if box != None:
                simulation.context.setPeriodicBoxVectors(box[0],box[1],box[2])
            # Compute the potential energy and append to list
            Energy = simulation.context.getState(getEnergy=True).getPotentialEnergy() / kilojoules_per_mole
            E.append(Energy)
            if dipole:
                D.append(get_dipole(simulation,q=q,positions=xyz1))

        print "\r",
        if verbose: print E
        if dipole:
            # Return a Nx4 array with energies in the first column and dipole in columns 2-4.
            return np.hstack((np.array(E).reshape(-1,1), np.array(D).reshape(-1,3)))
        else:
            return np.array(E)

Tinker_MD = None

def main():

    """
    Usage: (runcuda.sh) npt.py <openmm|gromacs|tinker> <liquid_prod_steps> <liquid_timestep (fs)> <liquid_interval (ps> <temperature> <pressure>

    This program is meant to be called automatically by ForceBalance on
    a GPU cluster (specifically, subroutines in openmmio.py).  It is
    not easy to use manually.  This is because the force field is read
    in from a ForceBalance 'FF' class.

    I wrote this program because automatic fitting of the density (or
    other equilibrium properties) is computationally intensive, and the
    calculations need to be distributed to the queue.  The main instance
    of ForceBalance (running on my workstation) queues up a bunch of these
    jobs (using Work Queue).  Then, I submit a bunch of workers to GPU
    clusters (e.g. Certainty, Keeneland).  The worker scripts connect to.
    the main instance and receives one of these jobs.

    This script can also be executed locally, if you want to (e.g. for
    debugging).  Just make sure you have the pickled 'forcebalance.p'
    file.

    """
    if args.engine == "openmm":
        if os.path.exists("liquid.pdb"):
            fnm = "liquid.pdb"
        elif os.path.exists("conf.pdb"):
            fnm = "conf.pdb"
    elif args.engine in ["gmx", "gromacs"]:
        fnm = "liquid.gro"
    elif args.engine == "tinker":
        fnm = "liquid.xyz"
    
    # Load in the molecule object so that we may count the number of molecules.
    try:
        M = Molecule(fnm)
    except:
        raise RuntimeError('Tried to load structure from %s but file does not exist' % fnm)
    
    # The number of molecules can be determined here.
    NMol = len(M.molecules)

    # Load the force field in from the ForceBalance pickle.
    FF,mvals,h,AGrad = lp_load(open('forcebalance.p'))
    FF.make(mvals)

    #=================================================================#
    # Run the simulation for the full system and analyze the results. #
    #=================================================================#
    EngineDict = {"openmm" : OpenMM_MD, "gmx" : Gromacs_MD,
                  "gromacs" : Gromacs_MD, "tinker" : Tinker_MD}

    # Create an instance of the MD Engine object.
    Engine = EngineDict[args.engine](FF)

    # This line runs the condensed phase simulation.
    Rhos, Potentials, Kinetics, Volumes, Dips, EDA = Engine.run_simulation("liquid", minimize=args.minimize_energy, savexyz=False)

    # Create a bunch of physical constants.
    # Energies are in kJ/mol
    # Lengths are in nanometers.
    L = len(Rhos)
    kB = 0.008314472471220214
    T = temperature
    kT = kB * T
    mBeta = -1.0 / kT
    Beta = 1.0 / kT
    atm_unit = 0.061019351687175
    bar_unit = 0.060221417930000
    # This is how I calculated the prefactor for the dielectric constant.
    # eps0 = 8.854187817620e-12 * coulomb**2 / newton / meter**2
    # epsunit = 1.0*(debye**2) / nanometer**3 / BOLTZMANN_CONSTANT_kB / kelvin
    # prefactor = epsunit/eps0/3
    prefactor = 30.348705333964077

    # Gather some physical variables.
    Energies = Potentials + Kinetics
    Ene_avg, Ene_err = mean_stderr(Energies)
    pV = atm_unit * pressure * Volumes
    pV_avg, pV_err = mean_stderr(pV)
    Rho_avg, Rho_err = mean_stderr(Rhos)
    PrintEDA(EDA, NMol)

    #==============================================#
    # Now run the simulation for just the monomer. #
    #==============================================#
    # Reset the timestep and such.
    global timestep, nsteps, nprod, nequil
    timestep = m_timestep
    nsteps   = m_nsteps
    nequil = m_nequil
    nprod = m_nprod

    # Run the OpenMM simulation, gather information.
    _, mPotentials, mKinetics, __, ___, mEDA = Engine.run_simulation("gas", minimize=args.minimize_energy, savexyz=False)
    mEnergies = mPotentials + mKinetics
    mEne_avg, mEne_err = mean_stderr(mEnergies)
    PrintEDA(mEDA, 1)

    #============================================#
    #  Compute the potential energy derivatives. #
    #============================================#
    print "Calculating potential energy derivatives with finite difference step size:", h
    # Switch for whether to compute the derivatives two different ways for consistency.
    FDCheck = False

    # Create a double-precision simulation object if desired (seems unnecessary).
    DoublePrecisionDerivatives = False
    if args.engine == "openmm" and DoublePrecisionDerivatives and AGrad:
        print "Creating Double Precision Simulation for parameter derivatives"
        Engine.create_simulation_object("liquid", precision="double")
        Engine.create_simulation_object("gas", precision="double")

    # Compute the energy and dipole derivatives.
    print "Condensed phase potential and dipole derivatives."
    print "Initializing array to length %i" % len(Energies)
    G, GDx, GDy, GDz = Engine.energy_derivatives(mvals, h, "liquid", len(Energies), AGrad, dipole=True)
    print "Gas phase potential derivatives."
    mG, _, __, ___ = Engine.energy_derivatives(mvals, h, "gas", len(mEnergies), AGrad, dipole=False)

    # Build the first density derivative.
    GRho = mBeta * (flat(np.mat(G) * col(Rhos)) / L - np.mean(Rhos) * np.mean(G, axis=1))

    # The enthalpy of vaporization in kJ/mol.
    Hvap_avg = mEne_avg - Ene_avg / NMol + kT - np.mean(pV) / NMol
    Hvap_err = np.sqrt(Ene_err**2 / NMol**2 + mEne_err**2 + pV_err**2/NMol**2)

    # Build the first Hvap derivative.
    GHvap = np.mean(G,axis=1)
    GHvap += mBeta * (flat(np.mat(G) * col(Energies)) / L - Ene_avg * np.mean(G, axis=1))
    GHvap /= NMol
    GHvap -= np.mean(mG,axis=1)
    GHvap -= mBeta * (flat(np.mat(mG) * col(mEnergies)) / L - mEne_avg * np.mean(mG, axis=1))
    GHvap *= -1
    GHvap -= mBeta * (flat(np.mat(G) * col(pV)) / L - np.mean(pV) * np.mean(G, axis=1)) / NMol

    # Print out the density and its derivative.
    Sep = printcool("Density: % .4f +- % .4f kg/m^3, Analytic Derivative" % (Rho_avg, Rho_err))
    FF.print_map(vals=GRho)
    print Sep

    H = Energies + pV
    # Print out the liquid enthalpy.
    print "Liquid enthalpy: % .4f kJ/mol, stdev % .4f ; (% .4f from energy, % .4f from pV)" % (np.mean(H), np.std(H), np.mean(Energies), np.mean(pV))
    V = np.array(Volumes)
    numboots = 1000

    def calc_rho(b = None, **kwargs):
        if b == None: b = np.ones(L,dtype=float)
        if 'r_' in kwargs:
            r_ = kwargs['r_']
        return bzavg(r_,b)

    # No need to calculate error using bootstrap, but here it is anyway
    # Rhoboot = []
    # for i in range(numboots):
    #    boot = np.random.randint(N,size=N)
    #    Rhoboot.append(calc_rho(None,**{'r_':Rhos[boot]}))
    # Rhoboot = np.array(Rhoboot)
    # Rho_err = np.std(Rhoboot)

    if FDCheck:
        Sep = printcool("Numerical Derivative:")
        GRho1 = property_derivatives(mvals, h, "liquid", kT, calc_rho, {'r_':Rhos}, Boxes)
        FF.print_map(vals=GRho1)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GRho, GRho1)]
        FF.print_map(vals=absfrac)

    print "Box total energy:", np.mean(Energies)
    print "Monomer total energy:", np.mean(mEnergies)
    Sep = printcool("Enthalpy of Vaporization: % .4f +- %.4f kJ/mol, Derivatives below" % (Hvap_avg, Hvap_err))
    FF.print_map(vals=GHvap)
    print Sep

    # Define some things to make the analytic derivatives easier.
    Gbar = np.mean(G,axis=1)
    def deprod(vec):
        return flat(np.mat(G)*col(vec))/L
    def covde(vec):
        return flat(np.mat(G)*col(vec))/L - Gbar*np.mean(vec)
    def avg(vec):
        return np.mean(vec)

    ## Thermal expansion coefficient and bootstrap error estimation
    def calc_alpha(b = None, **kwargs):
        if b == None: b = np.ones(L,dtype=float)
        if 'h_' in kwargs:
            h_ = kwargs['h_']
        if 'v_' in kwargs:
            v_ = kwargs['v_']
        return 1/(kT*T) * (bzavg(h_*v_,b)-bzavg(h_,b)*bzavg(v_,b))/bzavg(v_,b)
    Alpha = calc_alpha(None, **{'h_':H, 'v_':V})
    Alphaboot = []
    for i in range(numboots):
        boot = np.random.randint(L,size=L)
        Alphaboot.append(calc_alpha(None, **{'h_':H[boot], 'v_':V[boot]}))
    Alphaboot = np.array(Alphaboot)
    Alpha_err = np.std(Alphaboot) * max([np.sqrt(statisticalInefficiency(V)),np.sqrt(statisticalInefficiency(H))])

    ## Thermal expansion coefficient analytic derivative
    GAlpha1 = -1 * Beta * deprod(H*V) * avg(V) / avg(V)**2
    GAlpha2 = +1 * Beta * avg(H*V) * deprod(V) / avg(V)**2
    GAlpha3 = deprod(V)/avg(V) - Gbar
    GAlpha4 = Beta * covde(H)
    GAlpha  = (GAlpha1 + GAlpha2 + GAlpha3 + GAlpha4)/(kT*T)
    Sep = printcool("Thermal expansion coefficient: % .4e +- %.4e K^-1\nAnalytic Derivative:" % (Alpha, Alpha_err))
    FF.print_map(vals=GAlpha)
    if FDCheck:
        GAlpha_fd = property_derivatives(mvals, h, "liquid", kT, calc_alpha, {'h_':H,'v_':V}, Boxes)
        Sep = printcool("Numerical Derivative:")
        FF.print_map(vals=GAlpha_fd)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GAlpha, GAlpha_fd)]
        FF.print_map(vals=absfrac)

    ## Isothermal compressibility
    def calc_kappa(b=None, **kwargs):
        if b == None: b = np.ones(L,dtype=float)
        if 'v_' in kwargs:
            v_ = kwargs['v_']
        return bar_unit / kT * (bzavg(v_**2,b)-bzavg(v_,b)**2)/bzavg(v_,b)
    Kappa = calc_kappa(None,**{'v_':V})
    Kappaboot = []
    for i in range(numboots):
        boot = np.random.randint(L,size=L)
        Kappaboot.append(calc_kappa(None,**{'v_':V[boot]}))
    Kappaboot = np.array(Kappaboot)
    Kappa_err = np.std(Kappaboot) * np.sqrt(statisticalInefficiency(V))

    ## Isothermal compressibility analytic derivative
    Sep = printcool("Isothermal compressibility:    % .4e +- %.4e bar^-1\nAnalytic Derivative:" % (Kappa, Kappa_err))
    GKappa1 = +1 * Beta**2 * avg(V**2) * deprod(V) / avg(V)**2
    GKappa2 = -1 * Beta**2 * avg(V) * deprod(V**2) / avg(V)**2
    GKappa3 = +1 * Beta**2 * covde(V)
    GKappa  = bar_unit*(GKappa1 + GKappa2 + GKappa3)
    FF.print_map(vals=GKappa)
    if FDCheck:
        GKappa_fd = property_derivatives(mvals, h, "liquid", kT, calc_kappa, {'v_':V}, Boxes)
        Sep = printcool("Numerical Derivative:")
        FF.print_map(vals=GKappa_fd)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GKappa, GKappa_fd)]
        FF.print_map(vals=absfrac)

    ## Isobaric heat capacity
    def calc_cp(b=None, **kwargs):
        if b == None: b = np.ones(L,dtype=float)
        if 'h_' in kwargs:
            h_ = kwargs['h_']
        Cp_  = 1/(NMol*kT*T) * (bzavg(h_**2,b) - bzavg(h_,b)**2)
        Cp_ *= 1000 / 4.184
        return Cp_
    Cp = calc_cp(None,**{'h_':H})
    Cpboot = []
    for i in range(numboots):
        boot = np.random.randint(L,size=L)
        Cpboot.append(calc_cp(None,**{'h_':H[boot]}))
    Cpboot = np.array(Cpboot)
    Cp_err = np.std(Cpboot) * np.sqrt(statisticalInefficiency(H))

    ## Isobaric heat capacity analytic derivative
    GCp1 = 2*covde(H) * 1000 / 4.184 / (NMol*kT*T)
    GCp2 = mBeta*covde(H**2) * 1000 / 4.184 / (NMol*kT*T)
    GCp3 = 2*Beta*avg(H)*covde(H) * 1000 / 4.184 / (NMol*kT*T)
    GCp  = GCp1 + GCp2 + GCp3
    Sep = printcool("Isobaric heat capacity:        % .4e +- %.4e cal mol-1 K-1\nAnalytic Derivative:" % (Cp, Cp_err))
    FF.print_map(vals=GCp)
    if FDCheck:
        GCp_fd = property_derivatives(mvals, h, "liquid", kT, calc_cp, {'h_':H}, Boxes)
        Sep = printcool("Numerical Derivative:")
        FF.print_map(vals=GCp_fd)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GCp,GCp_fd)]
        FF.print_map(vals=absfrac)

    ## Dielectric constant
    def calc_eps0(b=None, **kwargs):
        if b == None: b = np.ones(L,dtype=float)
        if 'd_' in kwargs: # Dipole moment vector.
            d_ = kwargs['d_']
        if 'v_' in kwargs: # Volume.
            v_ = kwargs['v_']
        b0 = np.ones(L,dtype=float)
        dx = d_[:,0]
        dy = d_[:,1]
        dz = d_[:,2]
        D2  = bzavg(dx**2,b)-bzavg(dx,b)**2
        D2 += bzavg(dy**2,b)-bzavg(dy,b)**2
        D2 += bzavg(dz**2,b)-bzavg(dz,b)**2
        return prefactor*D2/bzavg(v_,b)/T
    Eps0 = calc_eps0(None,**{'d_':Dips, 'v_':V})
    Eps0boot = []
    for i in range(numboots):
        boot = np.random.randint(L,size=L)
        Eps0boot.append(calc_eps0(None,**{'d_':Dips[boot], 'v_':V[boot]}))
    Eps0boot = np.array(Eps0boot)
    Eps0_err = np.std(Eps0boot)*np.sqrt(np.mean([statisticalInefficiency(Dips[:,0]),statisticalInefficiency(Dips[:,1]),statisticalInefficiency(Dips[:,2])]))
 
    ## Dielectric constant analytic derivative
    Dx = Dips[:,0]
    Dy = Dips[:,1]
    Dz = Dips[:,2]
    D2 = avg(Dx**2)+avg(Dy**2)+avg(Dz**2)-avg(Dx)**2-avg(Dy)**2-avg(Dz)**2
    GD2  = 2*(flat(np.mat(GDx)*col(Dx))/L - avg(Dx)*(np.mean(GDx,axis=1))) - Beta*(covde(Dx**2) - 2*avg(Dx)*covde(Dx))
    GD2 += 2*(flat(np.mat(GDy)*col(Dy))/L - avg(Dy)*(np.mean(GDy,axis=1))) - Beta*(covde(Dy**2) - 2*avg(Dy)*covde(Dy))
    GD2 += 2*(flat(np.mat(GDz)*col(Dz))/L - avg(Dz)*(np.mean(GDz,axis=1))) - Beta*(covde(Dz**2) - 2*avg(Dz)*covde(Dz))
    GEps0 = prefactor*(GD2/avg(V) - mBeta*covde(V)*D2/avg(V)**2)/T
    Sep = printcool("Dielectric constant:           % .4e +- %.4e\nAnalytic Derivative:" % (Eps0, Eps0_err))
    FF.print_map(vals=GEps0)
    if FDCheck:
        GEps0_fd = property_derivatives(mvals, h, "liquid", kT, calc_eps0, {'d_':Dips,'v_':V}, Boxes)
        Sep = printcool("Numerical Derivative:")
        FF.print_map(vals=GEps0_fd)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GEps0,GEps0_fd)]
        FF.print_map(vals=absfrac)

    ## Print the final force field.
    pvals = FF.make(mvals)

    with open(os.path.join('npt_result.p'),'w') as f: lp_dump((Rhos, Volumes, Potentials, Energies, Dips, G, [GDx, GDy, GDz], mPotentials, mEnergies, mG, Rho_err, Hvap_err, Alpha_err, Kappa_err, Cp_err, Eps0_err, NMol),f)

if __name__ == "__main__":
    main()
