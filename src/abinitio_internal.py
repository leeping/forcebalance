""" @package forcebalance.abinitio_internal Internal implementation of energy matching (for TIP3P water only)

@author Lee-Ping Wang
@date 04/2012
"""

import os
from forcebalance import BaseReader
from forcebalance.abinitio import AbInitio
from forcebalance.forcefield import FF
import numpy as np
import sys
import pickle
import shutil
import itertools

class AbInitio_Internal(AbInitio):

    """Subclass of Target for force and energy matching
    using an internal implementation.  Implements the prepare and
    energy_force_driver methods.  The get method is in the superclass.

    The purpose of this class is to provide an extremely simple test
    case that does not require the user to install any external
    software.  It only runs with one of the included sample test
    calculations (internal_tip3p), and the objective function is
    energy matching.
    
    @warning This class is only intended to work with a very specific
    test case (internal_tip3p).  This is because the topology and
    ordering of the atoms is hard-coded (12 water molecules with 3
    atoms each).

    @warning This class does energy matching only (no forces)

    """

    def __init__(self,options,tgt_opts,forcefield):
        ## Name of the trajectory, we need this BEFORE initializing the SuperClass
        self.coords = "all.gro"
        ## Initialize the SuperClass!
        super(AbInitio_Internal,self).__init__(options,tgt_opts,forcefield)

    def energy_force_driver_all(self):
        """ Here we actually compute the interactions and return the
        energies and forces. I verified this to give the same answer
        as GROMACS. """

        M = []
        # Loop through the snapshots
        ThisFF = FF({'forcefield':['tip3p.xml'], 'ffdir':'', 'priors':{}},verbose=False)
        r_0   = ThisFF.pvals0[ThisFF.map['HarmonicBondForce.Bond/length/OW.HW']] * 10
        k_ij  = ThisFF.pvals0[ThisFF.map['HarmonicBondForce.Bond/k/OW.HW']]
        t_0   = ThisFF.pvals0[ThisFF.map['HarmonicAngleForce.Angle/angle/HW.OW.HW']] * 180 / np.pi
        k_ijk = ThisFF.pvals0[ThisFF.map['HarmonicAngleForce.Angle/k/HW.OW.HW']]
        q_o   = ThisFF.pvals0[ThisFF.map['NonbondedForce.Atom/charge/tip3p-O']]
        q_h   = ThisFF.pvals0[ThisFF.map['NonbondedForce.Atom/charge/tip3p-H']]
        sig   = ThisFF.pvals0[ThisFF.map['NonbondedForce.Atom/sigma/tip3p-O']]
        eps   = ThisFF.pvals0[ThisFF.map['NonbondedForce.Atom/epsilon/tip3p-O']]
        facel = 1389.35410

        for I in range(self.ns):
            xyz = self.mol.xyzs[I]
            Bond_Energy = 0.0
            Angle_Energy = 0.0
            VdW_Energy = 0.0
            Coulomb_Energy = 0.0
            for i in range(0,len(xyz),3):
                o = i
                h1 = i+1
                h2 = i+2
                # First O-H bond.
                r_1 = xyz[h1] - xyz[o]
                r_1n = np.linalg.norm(r_1)
                Bond_Energy += 0.5 * k_ij * ((r_1n - r_0) / 10)**2
                # Second O-H bond.
                r_2 = xyz[h2] - xyz[o]
                r_2n = np.linalg.norm(r_2)
                Bond_Energy += 0.5 * k_ij * ((r_2n - r_0) / 10)**2
                # Angle.
                theta = np.arccos(np.dot(r_1, r_2) / (r_1n * r_2n)) * 180 / np.pi
                Angle_Energy += 0.5 * k_ijk * ((theta - t_0) * np.pi / 180)**2
                for j in range(0, i, 3):
                    oo = j
                    hh1 = j+1
                    hh2 = j+2
                    # Lennard-Jones interaction.
                    r_o_oo = np.linalg.norm(xyz[oo] - xyz[o]) / 10
                    sroo = sig / r_o_oo
                    VdW_Energy += 4*eps*(sroo**12 - sroo**6)
                    # Coulomb interaction.
                    for k, l in itertools.product(*[[i,i+1,i+2],[j,j+1,j+2]]):
                        q1 = q_o if (k % 3 == 0) else q_h
                        q2 = q_o if (l % 3 == 0) else q_h
                        Coulomb_Energy += q1*q2 / np.linalg.norm(xyz[k]-xyz[l])
            Coulomb_Energy *= facel
            Energy = Bond_Energy + Angle_Energy + VdW_Energy + Coulomb_Energy
            Force = list(np.zeros(3*len(xyz)))
            M.append(np.array([Energy] + Force))
        return M
