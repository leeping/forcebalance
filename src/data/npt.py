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
from forcebalance.nifty import col, flat, lp_dump, lp_load, printcool, printcool_dictionary, statisticalInefficiency, which, _exec, isint, wopen
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, f1d7p, in_fd
from forcebalance.molecule import Molecule
from forcebalance.output import getLogger
logger = getLogger(__name__)

#========================================================#
#| Global, user-tunable variables (simulation settings) |#
#========================================================#

# TODO: Strip out PDB and XML file arguments.
# TODO: Strip out all units.

parser = argparse.ArgumentParser()
parser.add_argument('engine', help='MD program that we are using; choose "openmm" or "gromacs"')
parser.add_argument('liquid_nsteps', type=int, help='Number of time steps for the liquid production simulation')
parser.add_argument('liquid_timestep', type=float, help='Length of the time step for the liquid simulation, in femtoseconds')
parser.add_argument('liquid_intvl', type=float, help='Time interval for saving the liquid coordinates, in picoseconds')
parser.add_argument('temperature',type=float, help='Temperature (K)')
parser.add_argument('pressure',type=float, help='Pressure (Atm)')

# Other optional arguments
parser.add_argument('--liquid_nequil', type=int, help='Number of time steps used for equilibration', default=100000)
parser.add_argument('--gas_nsteps', type=int, help='Number of time steps for the gas-phase production simulation', default=1000000)
parser.add_argument('--gas_timestep', type=float, help='Time step for the gas-phase simulation, in femtoseconds', default=0.5)
parser.add_argument('--gas_nequil', type=int, help='Number of time steps for the gas-phase equilibration simulation', default=100000)
parser.add_argument('--gas_intvl', type=float, help='Time interval for saving the gas-phase coordinates, in picoseconds', default=0.1)
parser.add_argument('--anisotropic', action='store_true', help='Enable anisotropic scaling of periodic box (useful for crystals)')
parser.add_argument('--mts', action='store_true', help='Enable multiple timestep integrator (OpenMM)')
parser.add_argument('--nt', type=int, help='Number of threads when executing GROMACS', default=1)
parser.add_argument('--force_cuda', action='store_true', help='Exit if CUDA platform is not available (OpenMM)')
parser.add_argument('--gmxpath', type=str, help='Specify the location of GROMACS executables', default="")
parser.add_argument('--minimize', action='store_true', help='Minimize the energy of the system prior to running dynamics')

args = parser.parse_args()

faststep         = 0.25                                                        # "fast" timestep (for MTS integrator, if used)
temperature      = args.temperature                                            # temperature in kelvin
pressure         = args.pressure                                               # pressure in atmospheres

if args.engine.lower() == "openmm":
    try:
        from simtk.unit import *
        from simtk.openmm import *
        from simtk.openmm.app import *
        from forcebalance.openmmio import *
    except:
        traceback.print_exc()
        raise Exception("Cannot import OpenMM modules")
    Engine = OpenMM
elif args.engine.lower() == "gromacs" or args.engine == "gmx":
    from forcebalance.gmxio import *
    if args.mts:
        raise Exception("Multiple timestep integrator is only usable with OpenMM interface")
    if args.force_cuda:
        raise Exception("CUDA platform is only usable with OpenMM interface")
    Engine = GMX
elif args.engine.lower() == "tinker":
    from forcebalance.tinkerio import *
    if args.mts:
        raise Exception("Multiple timestep integrator it is only usable with OpenMM interface")
    if args.force_cuda:
        raise Exception("CUDA platform is only usable with OpenMM interface")
    Engine = TINKER
else:
    raise Exception('Only OpenMM and GROMACS support implemented at this time.')

printcool("ForceBalance condensed phase simulation using engine: %s" % args.engine.upper(), color=4, bold=True)

liquid_snapshots = (args.liquid_nsteps * args.liquid_timestep / 1000) / args.liquid_intvl
liquid_iframes = 1000 * args.liquid_intvl / args.liquid_timestep
gas_snapshots = (args.gas_nsteps * args.gas_timestep / 1000) / args.gas_intvl
gas_iframes = 1000 * args.gas_intvl / args.gas_timestep

print "For the condensed phase system, I will collect %i snapshots spaced apart by %i x %.3f fs time steps" \
    % (liquid_snapshots, liquid_iframes, args.liquid_timestep)
print "For the gas phase system, I will collect %i snapshots spaced apart by %i x %.3f fs time steps" \
    % (gas_snapshots, gas_iframes, args.gas_timestep)
if liquid_snapshots < 2:
    raise Exception('Please set the number of liquid time steps so that you collect at least two snapshots (minimum %i)' \
                        % (2000 * (args.liquid_intvl/args.liquid_timestep)))
if gas_snapshots < 2:
    raise Exception('Please set the number of gas time steps so that you collect at least two snapshots (minimum %i)' \
                        % (2000 * (args.gas_intvl/args.gas_timestep)))

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
    printcool_dictionary(PrintDict, "Energy Component Analysis, Mean +- Stderr [Per Molecule] (kJ/mol)")

#=============================================#
#|   Functions for differentiating energy    |#
#|            and properties                 |#
#=============================================#

def energy_derivatives(engine, FF, mvals, h, length, AGrad=True, dipole=False):

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
    G        = np.zeros((FF.np,length))
    GDx      = np.zeros((FF.np,length))
    GDy      = np.zeros((FF.np,length))
    GDz      = np.zeros((FF.np,length))
    if not AGrad:
        return G, GDx, GDy, GDz
    def energy_driver(mvals_):
        FF.make(mvals_)
        if dipole:
            return engine.energy_dipole()
        else:
            return engine.energy()

    ED0      = energy_driver(mvals)
    for i in range(FF.np):
        print i, FF.plist[i] + " "*30 + "\r",
        EDG, _   = f12d3p(fdwrap(energy_driver,mvals,i),h,f0=ED0)
        if dipole:
            G[i,:]   = EDG[:,0]
            GDx[i,:] = EDG[:,1]
            GDy[i,:] = EDG[:,2]
            GDz[i,:] = EDG[:,3]
        else:
            G[i,:]   = EDG[:]
    return G, GDx, GDy, GDz

def property_derivatives(engine, FF, mvals, h, kT, property_driver, property_kwargs, AGrad=True):

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
    G        = np.zeros(FF.np)
    if not AGrad:
        return G
    def energy_driver(mvals_):
        FF.make(mvals_)
        return engine.energy_dipole()

    ED0      = energy_driver(mvals)
    E0       = ED0[:,0]
    D0       = ED0[:,1:]
    P0       = property_driver(None, **property_kwargs)
    if 'h_' in property_kwargs:
        H0 = property_kwargs['h_'].copy()
    for i in range(FF.np):
        print FF.plist[i] + " "*30
        ED1      = fdwrap(energy_driver,mvals,i)(h)
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
        EDM1      = fdwrap(energy_driver,mvals,i)(-h)
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
    
def main():

    """
    Usage: (runcuda.sh) npt.py <openmm|gromacs|tinker> <liquid_nsteps> <liquid_timestep (fs)> <liquid_intvl (ps> <temperature> <pressure>

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
    
    # Find either a coordinate file that matches the pattern "liquid.gro|xyz|pdb"
    # or a single coordinate file that contains multiple molecules.
    allfnms = [i for i in os.listdir('.') if (os.path.isfile(i) and any([i.endswith(j) for j in ".gro", ".pdb", ".xyz"]))]
    liqfnms = ["liquid.gro", "liquid.xyz", "liquid.pdb"]
    if sum([os.path.exists(i) for i in liqfnms]) == 1:
        liqfnm = liqfnms[[os.path.exists(i) for i in liqfnms].index(True)]
    else:
        nas = [len(Molecule(i).molecules) > 1 for i in allfnms]
        if sum(nas) > 1:
            raise IOError('Cannot determine the liquid coordinate file.')
        liqfnm = allfnms[nas.index(True)]
    ML = Molecule(liqfnm)

    # Find either a coordinate file that matches the pattern "gas.gro|xyz|pdb"
    # or a single coordinate file that contains one molecules.
    gasfnms = ["gas.gro", "gas.xyz", "gas.pdb"]
    if sum([os.path.exists(i) for i in gasfnms]) == 1:
        gasfnm = gasfnms[[os.path.exists(i) for i in gasfnms].index(True)]
    else:
        nas = [len(Molecule(i).molecules) == 1 for i in allfnms]
        if sum(nas) > 1:
            raise IOError('Cannot determine the gas coordinate file.')
        gasfnm = allfnms[nas.index(True)]
    MG = Molecule(gasfnm)
    
    # The number of molecules can be determined here.
    NMol = len(ML.molecules)

    # Load the force field in from the ForceBalance pickle.
    FF,mvals,h,AGrad = lp_load(open('forcebalance.p'))
    FF.make(mvals)

    ## Setting of options.
    ## Engine options
    EngOpts = OrderedDict()
    EngOpts["liquid"] = OrderedDict([("coords", liqfnm), ("mol", ML), ("pbc", True)])
    EngOpts["gas"] = OrderedDict([("coords", gasfnm), ("mol", MG), ("pbc", False)])
    GenOpts = OrderedDict()
    if args.engine == "openmm":
        GenOpts["ffxml"] = FF.openmmxml
    elif args.engine == "gromacs":
        EngOpts["liquid"]["gmx_top"] = os.path.splitext(liqfnm)[0] + ".top"
        EngOpts["liquid"]["gmx_mdp"] = os.path.splitext(liqfnm)[0] + ".mdp"
        EngOpts["gas"]["gmx_top"] = os.path.splitext(gasfnm)[0] + ".top"
        EngOpts["gas"]["gmx_mdp"] = os.path.splitext(gasfnm)[0] + ".mdp"
        if args.gmxpath != '': GenOpts["gmxpath"] = args.gmxpath
    elif args.engine == "tinker":
        EngOpts["liquid"]["tinker_key"] = os.path.splitext(liqfnm)[0] + ".key"
        EngOpts["gas"]["tinker_key"] = os.path.splitext(gasfnm)[0] + ".key"
        GenOpts["tinkerprm"] = FF.tinkerprm
    EngOpts["liquid"].update(GenOpts)
    EngOpts["gas"].update(GenOpts)
    for i in EngOpts:
        printcool_dictionary(EngOpts[i], "Engine options for %s" % i)
    ## Molecular dynamics options
    MDOpts = OrderedDict()
    MDOpts["liquid"] = OrderedDict([("nsteps", args.liquid_nsteps), ("timestep", args.liquid_timestep),
                                    ("temperature", temperature), ("pressure", pressure),
                                    ("nequil", args.liquid_nequil), ("minimize", args.minimize),
                                    ("nsave", int(1000 * args.liquid_intvl / args.liquid_timestep)),
                                    ("threads", args.nt), ("verbose", True)])
    
    MDOpts["gas"] = OrderedDict([("nsteps", args.gas_nsteps), ("timestep", args.gas_timestep),
                                 ("temperature", temperature), ("nsave", int(1000 * args.gas_intvl / args.gas_timestep)),
                                 ("nequil", args.gas_nequil), ("minimize", args.minimize), ("threads", 1)])

    DoEDA = not (args.engine == "openmm" and args.mts)

    #=================================================================#
    # Run the simulation for the full system and analyze the results. #
    #=================================================================#
    # Create an instance of the MD Engine object.
    Liquid = Engine(name="liquid", **EngOpts["liquid"])
    Gas = Engine(name="gas", **EngOpts["gas"])

    # This line runs the condensed phase simulation.
    printcool("Condensed phase molecular dynamics", color=4, bold=True)
    Rhos, Potentials, Kinetics, Volumes, Dips, EDA = \
        Liquid.molecular_dynamics(**MDOpts["liquid"])

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
    if DoEDA: PrintEDA(EDA, NMol)

    #==============================================#
    # Now run the simulation for just the monomer. #
    #==============================================#

    # Run the OpenMM simulation, gather information.

    printcool("Gas phase molecular dynamics", color=4, bold=True)
    _, mPotentials, mKinetics, __, ___, mEDA = \
        Gas.molecular_dynamics(**MDOpts["gas"])

    mEnergies = mPotentials + mKinetics
    mEne_avg, mEne_err = mean_stderr(mEnergies)
    if DoEDA: PrintEDA(mEDA, 1)

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
        Liquid = Engine(name="liquid", openmm_precision="double", **EngOpts["liquid"])
        Gas = Engine(name="gas", openmm_precision="double", **EngOpts["gas"])

    # Compute the energy and dipole derivatives.
    printcool("Condensed phase energy and dipole derivatives\nInitializing array to length %i" % len(Energies), color=4, bold=True)
    G, GDx, GDy, GDz = energy_derivatives(Liquid, FF, mvals, h, len(Energies), AGrad, dipole=True)
    printcool("Gas phase energy derivatives", color=4, bold=True)
    mG, _, __, ___ = energy_derivatives(Gas, FF, mvals, h, len(mEnergies), AGrad, dipole=False)

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
    Sep = printcool("Density: % .4f +- % .4f kg/m^3\nAnalytic Derivative:" % (Rho_avg, Rho_err))
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
        GRho1 = property_derivatives(Liquid, FF, mvals, h, kT, calc_rho, {'r_':Rhos})
        FF.print_map(vals=GRho1)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GRho, GRho1)]
        FF.print_map(vals=absfrac)

    print "Box total energy:", np.mean(Energies)
    print "Monomer total energy:", np.mean(mEnergies)
    Sep = printcool("Enthalpy of Vaporization: % .4f +- %.4f kJ/mol\nAnalytic Derivative:" % (Hvap_avg, Hvap_err))
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
        GAlpha_fd = property_derivatives(Liquid, FF, mvals, h, kT, calc_alpha, {'h_':H,'v_':V})
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
    Sep = printcool("Isothermal compressibility:  % .4e +- %.4e bar^-1\nAnalytic Derivative:" % (Kappa, Kappa_err))
    GKappa1 = +1 * Beta**2 * avg(V**2) * deprod(V) / avg(V)**2
    GKappa2 = -1 * Beta**2 * avg(V) * deprod(V**2) / avg(V)**2
    GKappa3 = +1 * Beta**2 * covde(V)
    GKappa  = bar_unit*(GKappa1 + GKappa2 + GKappa3)
    FF.print_map(vals=GKappa)
    if FDCheck:
        GKappa_fd = property_derivatives(Liquid, FF, mvals, h, kT, calc_kappa, {'v_':V})
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
    Sep = printcool("Isobaric heat capacity:  % .4e +- %.4e cal mol-1 K-1\nAnalytic Derivative:" % (Cp, Cp_err))
    FF.print_map(vals=GCp)
    if FDCheck:
        GCp_fd = property_derivatives(Liquid, FF, mvals, h, kT, calc_cp, {'h_':H})
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
        GEps0_fd = property_derivatives(Liquid, FF, mvals, h, kT, calc_eps0, {'d_':Dips,'v_':V})
        Sep = printcool("Numerical Derivative:")
        FF.print_map(vals=GEps0_fd)
        Sep = printcool("Difference (Absolute, Fractional):")
        absfrac = ["% .4e  % .4e" % (i-j, (i-j)/j) for i,j in zip(GEps0,GEps0_fd)]
        FF.print_map(vals=absfrac)

    ## Print the final force field.
    pvals = FF.make(mvals)

    with wopen(os.path.join('npt_result.p')) as f: lp_dump((Rhos, Volumes, Potentials, Energies, Dips, G, [GDx, GDy, GDz], mPotentials, mEnergies, mG, Rho_err, Hvap_err, Alpha_err, Kappa_err, Cp_err, Eps0_err, NMol),f)

if __name__ == "__main__":
    main()
