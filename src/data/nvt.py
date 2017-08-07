#!/usr/bin/env python
#--- coding: utf-8 ---

"""
@package nvt

Runs a simulation to compute condensed phase properties in NVT ensemble
(for example, the surface tension) and compute the derivative with respect
to changing the force field parameters. This script is a part of ForceBalance.

The algorithm used to compute the surface tension is the test-area method.
We run a NVT simulation on a film of water in a long PBC box. For each frame, we
increase and decrease the area of the xy surface by ΔS but keep the volumn constant,
and compute the energy change ΔU+ and UΔ-, then the surface tension is computed by:

γ = -kT/(2ΔS) * [ ln<exp(-ΔE+/kT)> - ln<exp(-ΔE-/kT)> ]

References

[1] Vega C, Miguel E. de. Surface tension of the most popular models of water by
using the test-area simulation method JCP 126:154707, 2007.

The partial derivatives of surface tension with respect to forcefield parameters
are computed using the analytic formula:

β = 1/kT
∂γ/∂α = -kT/(2ΔS) * { 1/<exp(-βΔE+)> * [<-β ∂E+/∂α exp(-βΔE+)> - <exp(-βΔE+)><-β ∂E/∂α>]
                     -1/<exp(-βΔE-)> * [<-β ∂E-/∂α exp(-βΔE-)> - <exp(-βΔE-)><-β ∂E/∂α>] }

Copyright And License

@author Yudong Qiu <qiu@ucdavis.edu>
@author Lee-Ping Wang <leeping@ucdavis.edu>

Note: Many parts of this script is directly copied from npt.py

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
from collections import namedtuple, OrderedDict
from forcebalance.forcefield import FF
from forcebalance.nifty import col, flat, lp_dump, lp_load, printcool, printcool_dictionary, statisticalInefficiency, which, _exec, isint, wopen, click
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, f1d7p, in_fd
from forcebalance.molecule import Molecule
from forcebalance.output import getLogger
logger = getLogger(__name__)

#========================================================#
#| Global, user-tunable variables (simulation settings) |#
#========================================================#
parser = argparse.ArgumentParser("nvt simulation tool for computing surface tension and derivatives.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('engine', choices=['openmm'], help='MD program that we are using.')
parser.add_argument('temperature', type=float, help='Temperature (K)')

args = parser.parse_args()

faststep         = 0.25                    # "fast" timestep (for MTS integrator, if used)
temperature      = args.temperature        # temperature in kelvin
engname          = args.engine.lower()     # Name of the engine

if engname == "openmm":
    try:
        from simtk.unit import *
        from simtk.openmm import *
        from simtk.openmm.app import *
    except:
        traceback.print_exc()
        raise Exception("Cannot import OpenMM modules")
    from forcebalance.openmmio import *
    Engine = OpenMM
#elif engname == "gromacs" or engname == "gmx":
#    from forcebalance.gmxio import *
#    Engine = GMX
#elif engname == "tinker":
#    from forcebalance.tinkerio import *
#    Engine = TINKER
else:
    raise Exception('Only OpenMM is supported at this time.')

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

def energy_derivatives(engine, FF, mvals, h, pgrad, length, AGrad=True, dipole=False):

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
    for i in pgrad:
        logger.info("%i %s\r" % (i, (FF.plist[i] + " "*30)))
        EDG, _   = f12d3p(fdwrap(energy_driver,mvals,i),h,f0=ED0)
        if dipole:
            G[i,:]   = EDG[:,0]
            GDx[i,:] = EDG[:,1]
            GDy[i,:] = EDG[:,2]
            GDz[i,:] = EDG[:,3]
        else:
            G[i,:]   = EDG[:]
    #reset FF parameters
    FF.make(mvals)
    return G, GDx, GDy, GDz

def property_derivatives(engine, FF, mvals, h, pgrad, kT, property_driver, property_kwargs, AGrad=True):

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
    for i in pgrad:
        logger.info("%s\n" % (FF.plist[i] + " "*30))
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
            logger.warn("Warning: Effective number of snapshots: % .1f (out of %i)\n" % (InfoContent, len(E0)))
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
            logger.warn("Warning: Effective number of snapshots: % .1f (out of %i)\n" % (InfoContent, len(E0)))
        PM1 = property_driver(b=b,**property_kwargs)
        G[i] = (P1-PM1)/(2*h)
    if 'h_' in property_kwargs:
        property_kwargs['h_'] = H0.copy()
    if 'd_' in property_kwargs:
        property_kwargs['d_'] = D0.copy()
    return G

def main():

    """
    Usage: (runcuda.sh) nvt.py <openmm|gromacs|tinker> <liquid_nsteps> <liquid_timestep (fs)> <liquid_intvl (ps> <temperature>

    This program is meant to be called automatically by ForceBalance on
    a GPU cluster (specifically, subroutines in openmmio.py).  It is
    not easy to use manually.  This is because the force field is read
    in from a ForceBalance 'FF' class.
    """

    printcool("ForceBalance condensed phase NVT simulation using engine: %s" % engname.upper(), color=4, bold=True)

    #----
    # Load the ForceBalance pickle file which contains:
    #----
    # - Force field object
    # - Optimization parameters
    # - Options from the Target object that launched this simulation
    # - Switch for whether to evaluate analytic derivatives.
    FF,mvals,TgtOptions,AGrad = lp_load('forcebalance.p')
    FF.ffdir = '.'
    # Write the force field file.
    FF.make(mvals)

    #----
    # Load the options that are set in the ForceBalance input file.
    #----
    # Finite difference step size
    h = TgtOptions['h']
    pgrad = TgtOptions['pgrad']
    # MD options; time step (fs), production steps, equilibration steps, interval for saving data (ps)
    nvt_timestep = TgtOptions['nvt_timestep']
    nvt_md_steps = TgtOptions['nvt_md_steps']
    nvt_eq_steps = TgtOptions['nvt_eq_steps']
    nvt_interval = TgtOptions['nvt_interval']
    liquid_fnm = TgtOptions['nvt_coords']

    # Number of threads, multiple timestep integrator, anisotropic box etc.
    threads = TgtOptions.get('md_threads', 1)
    mts = TgtOptions.get('mts_integrator', 0)
    rpmd_beads = TgtOptions.get('rpmd_beads', 0)
    force_cuda = TgtOptions.get('force_cuda', 0)
    nbarostat = TgtOptions.get('n_mcbarostat', 25)
    anisotropic = TgtOptions.get('anisotropic_box', 0)
    minimize = TgtOptions.get('minimize_energy', 1)

    # Print all options.
    printcool_dictionary(TgtOptions, title="Options from ForceBalance")
    nvt_snapshots = (nvt_timestep * nvt_md_steps / 1000) / nvt_interval
    nvt_iframes = 1000 * nvt_interval / nvt_timestep
    logger.info("For the condensed phase system, I will collect %i snapshots spaced apart by %i x %.3f fs time steps\n" \
        % (nvt_snapshots, nvt_iframes, nvt_timestep))
    if nvt_snapshots < 2:
        raise Exception('Please set the number of liquid time steps so that you collect at least two snapshots (minimum %i)' \
                            % (2000 * (nvt_interval/nvt_timestep)))

    #----
    # Loading coordinates
    #----
    ML = Molecule(liquid_fnm, toppbc=True)
    # Determine the number of molecules in the condensed phase coordinate file.
    NMol = len(ML.molecules)
    TgtOptions['n_molecules'] = NMol
    logger.info("There are %i molecules in the liquid\n" % (NMol))

    #----
    # Setting up MD simulations
    #----
    EngOpts = OrderedDict()
    EngOpts["liquid"] = OrderedDict([("coords", liquid_fnm), ("mol", ML), ("pbc", True)])
    if "nonbonded_cutoff" in TgtOptions:
        EngOpts["liquid"]["nonbonded_cutoff"] = TgtOptions["nonbonded_cutoff"]
    else:
        largest_available_cutoff = min(ML.boxes[0][:3]) / 2 - 0.1
        EngOpts["liquid"]["nonbonded_cutoff"] = largest_available_cutoff
        logger.info("nonbonded_cutoff is by default set to the largest available value: %g Angstrom" %largest_available_cutoff)
    if "vdw_cutoff" in TgtOptions:
        EngOpts["liquid"]["vdw_cutoff"] = TgtOptions["vdw_cutoff"]
    # Hard Code nonbonded_cutoff to 13A for test
    #EngOpts["liquid"]["nonbonded_cutoff"] = EngOpts["liquid"]["vdw_cutoff"] = 13.0
    GenOpts = OrderedDict([('FF', FF)])
    if engname == "openmm":
        # OpenMM-specific options
        EngOpts["liquid"]["platname"] = TgtOptions.get("platname", 'CUDA')
        if force_cuda:
            try: Platform.getPlatformByName('CUDA')
            except: raise RuntimeError('Forcing failure because CUDA platform unavailable')
            EngOpts["liquid"]["platname"] = 'CUDA'
        if threads > 1: logger.warn("Setting the number of threads will have no effect on OpenMM engine.\n")

    EngOpts["liquid"].update(GenOpts)
    for i in EngOpts:
        printcool_dictionary(EngOpts[i], "Engine options for %s" % i)

    # Set up MD options
    MDOpts = OrderedDict()
    MDOpts["liquid"] = OrderedDict([("nsteps", nvt_md_steps), ("timestep", nvt_timestep),
                                    ("temperature", temperature),
                                    ("nequil", nvt_eq_steps), ("minimize", minimize),
                                    ("nsave", int(1000 * nvt_interval / nvt_timestep)),
                                    ("verbose", True),
                                    ('save_traj', TgtOptions['save_traj']),
                                    ("threads", threads),
                                    ("mts", mts), ("rpmd_beads", rpmd_beads), ("faststep", faststep)])

    # Energy components analysis disabled for OpenMM MTS because it uses force groups
    if (engname == "openmm" and mts): logger.warn("OpenMM with MTS integrator; energy components analysis will be disabled.\n")

    # Create instances of the MD Engine objects.
    Liquid = Engine(name="liquid", **EngOpts["liquid"])

    #=================================================================#
    # Run the simulation for the full system and analyze the results. #
    #=================================================================#

    printcool("Condensed phase NVT molecular dynamics", color=4, bold=True)
    click()
    prop_return = Liquid.molecular_dynamics(**MDOpts["liquid"])
    logger.info("Liquid phase MD simulation took %.3f seconds\n" % click())
    Potentials = prop_return['Potentials']

    #============================================#
    #  Compute the potential energy derivatives. #
    #============================================#
    if AGrad:
        logger.info("Calculating potential energy derivatives with finite difference step size: %f\n" % h)
        # Switch for whether to compute the derivatives two different ways for consistency.
        FDCheck = False
        printcool("Condensed phase energy and dipole derivatives\nInitializing array to length %i" % len(Potentials), color=4, bold=True)
        click()
        G, GDx, GDy, GDz = energy_derivatives(Liquid, FF, mvals, h, pgrad, len(Potentials), AGrad, dipole=False)
        logger.info("Condensed phase energy derivatives took %.3f seconds\n" % click())

    #==============================================#
    #  Condensed phase properties and derivatives. #
    #==============================================#

    # Physical constants
    kB = 0.008314472471220214
    T = temperature
    kT = kB * T # Unit: kJ/mol

    #--- Surface Tension ----
    logger.info("Start Computing surface tension.\n")
    perturb_proportion = 0.0005
    box_vectors = np.array(Liquid.xyz_omms[0][1]/nanometer) # obtain original box vectors from first frame
    delta_S = np.sqrt(np.sum(np.cross(box_vectors[0], box_vectors[1])**2)) * perturb_proportion * 2 # unit: nm^2. *2 for 2 surfaces
    # perturb xy area +
    click()
    scale_x = scale_y = np.sqrt(1 + perturb_proportion)
    scale_z = 1.0 / (1+perturb_proportion) # keep the box volumn while changing the area of xy plane
    Liquid.scale_box(scale_x, scale_y, scale_z)
    logger.info("scale_box+ took %.3f seconds\n" %click())
    # Obtain energies and gradients
    Potentials_plus = Liquid.energy()
    logger.info("Calculation of energies for perturbed box+ took %.3f seconds\n" %click())
    if AGrad:
        G_plus, _, _, _ = energy_derivatives(Liquid, FF, mvals, h, pgrad, len(Potentials), AGrad, dipole=False)
        logger.info("Calculation of energy gradients for perturbed box+ took %.3f seconds\n" %click())
    # perturb xy area - ( Note: also need to cancel the previous scaling)
    scale_x = scale_y = np.sqrt(1 - perturb_proportion) * (1.0/scale_x)
    scale_z = 1.0 / (1-perturb_proportion) * (1.0/scale_z)
    Liquid.scale_box(scale_x, scale_y, scale_z)
    logger.info("scale_box- took %.3f seconds\n" %click())
    # Obtain energies and gradients
    Potentials_minus = Liquid.energy()
    logger.info("Calculation of energies for perturbed box- took %.3f seconds\n" %click())
    if AGrad:
        G_minus, _, _, _ = energy_derivatives(Liquid, FF, mvals, h, pgrad, len(Potentials), AGrad, dipole=False)
        logger.info("Calculation of energy gradients for perturbed box- took %.3f seconds\n" %click())
    # Compute surface tension
    dE_plus = Potentials_plus - Potentials # Unit: kJ/mol
    dE_minus = Potentials_minus - Potentials # Unit: kJ/mol
    prefactor = -0.5 * kT / delta_S / 6.0221409e-1 # Unit mJ m^-2
    # Following equation: γ = -kT/(2ΔS) * [ ln<exp(-ΔE+/kT)> - ln<exp(-ΔE-/kT)> ]
    #plus_avg, plus_err = mean_stderr(np.exp(-dE_plus/kT))
    #minus_avg, minus_err = mean_stderr(np.exp(-dE_minus/kT))
    #surf_ten = -0.5 * kT / delta_S * ( np.log(plus_avg) - np.log(minus_avg) ) / 6.0221409e-1 # convert to mJ m^-2
    #surf_ten_err = 0.5 * kT / delta_S * ( np.log(plus_avg+plus_err) - np.log(plus_avg-plus_err) + np.log(minus_avg+minus_err) - np.log(minus_avg-minus_err) ) / 6.0221409e-1
    exp_dE_plus = np.exp(-dE_plus/kT)
    exp_dE_minus = np.exp(-dE_minus/kT)
    surf_ten = prefactor * ( np.log(np.mean(exp_dE_plus)) - np.log(np.mean(exp_dE_minus)) )
    # Use bootstrap method to estimate the error
    num_frames = len(exp_dE_plus)
    numboots = 1000
    surf_ten_boots = np.zeros(numboots)
    for i in xrange(numboots):
        boots_ordering = np.random.randint(num_frames, size=num_frames)
        boots_exp_dE_plus = np.take(exp_dE_plus, boots_ordering)
        boots_exp_dE_minus = np.take(exp_dE_minus, boots_ordering)
        surf_ten_boots[i] = prefactor * ( np.log(np.mean(boots_exp_dE_plus)) - np.log(np.mean(boots_exp_dE_minus)) )
    surf_ten_err = np.std(surf_ten_boots) * np.sqrt(np.mean([statisticalInefficiency(exp_dE_plus), statisticalInefficiency(exp_dE_minus)]))

    printcool("Surface Tension:       % .4f +- %.4f mJ m^-2" % (surf_ten, surf_ten_err))
    # Analytic Gradient of surface tension
    # Formula:      β = 1/kT
    #           ∂γ/∂α = -kT/(2ΔS) * { 1/<exp(-βΔE+)> * [<-β ∂E+/∂α exp(-βΔE+)> - <-β ∂E/∂α><exp(-βΔE+)>]
    #                                -1/<exp(-βΔE-)> * [<-β ∂E-/∂α exp(-βΔE-)> - <-β ∂E/∂α><exp(-βΔE-)>] }
    n_params = len(mvals)
    G_surf_ten = np.zeros(n_params)
    if AGrad:
        beta = 1.0 / kT
        plus_denom = np.mean(np.exp(-beta*dE_plus))
        minus_denom = np.mean(np.exp(-beta*dE_minus))
        for param_i in xrange(n_params):
            plus_left = np.mean(-beta * G_plus[param_i] * np.exp(-beta*dE_plus))
            plus_right = np.mean(-beta * G[param_i]) * plus_denom
            minus_left = np.mean(-beta * G_minus[param_i] * np.exp(-beta*dE_minus))
            minus_right = np.mean(-beta * G[param_i]) * minus_denom
            G_surf_ten[param_i] = prefactor * (1.0/plus_denom*(plus_left-plus_right) - 1.0/minus_denom*(minus_left-minus_right))
        printcool("Analytic Derivatives:")
        FF.print_map(vals=G_surf_ten)

    logger.info("Writing final force field.\n")
    pvals = FF.make(mvals)

    logger.info("Writing all results to disk.\n")
    result_dict = {'surf_ten': surf_ten, 'surf_ten_err': surf_ten_err, 'G_surf_ten': G_surf_ten}
    lp_dump(result_dict, 'nvt_result.p')

if __name__ == "__main__":
    main()
