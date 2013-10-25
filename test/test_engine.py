import unittest
import sys, os, re
import forcebalance
import abc
import numpy as np
from __init__ import ForceBalanceTestCase
from forcebalance.nifty import *
from forcebalance.tinkerio import TINKER
from forcebalance.openmmio import OpenMM
from test_target import TargetTests # general targets tests defined in test_target.py

class TestOpenMM_vs_TINKER(ForceBalanceTestCase):
    def setUp(self):
        self.logger.debug("\nBuilding options for target...\n")
        self.cwd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), "test", "openmm_vs_tinker"))
        if not os.path.exists("temp"): os.makedirs("temp")
        os.chdir("temp")
        os.system("ln -s ../prism.pdb")
        os.system("ln -s ../prism.key")
        os.system("ln -s ../hex.arc")
        os.system("ln -s ../water.prm")
        os.system("ln -s ../amoebawater.xml")
        os.chdir("..")
        self.addCleanup(os.system, 'rm -rf temp')

    def test_energy_force(self):
        """ Compare OpenMM vs. TINKER energy and forces with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER energy and force with AMOEBA")
        os.chdir("temp")
        tinkerpath=which('testgrad')
        if (which('testgrad') == ''):
            self.skipTest("TINKER programs are not in the PATH.")
        O = OpenMM(coords="hex.arc", pdb="prism.pdb", ffxml="amoebawater.xml", precision="double", \
               mmopts={'rigidWater':False, 'mutualInducedTargetEpsilon':1e-6})
        T = TINKER(coords="hex.arc", tinker_key="prism.key", tinkerpath=tinkerpath)
        EF_O = O.energy_force()[0]
        EF_T = T.energy_force()[0]
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same AMOEBA energy to within 0.001 kJ\n")
        self.assertAlmostEqual(EF_O[0], EF_T[0], msg="OpenMM and TINKER energies are different", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same AMOEBA energy to within 0.01 kJ/mol/nm\n")
        self.assertNdArrayEqual(EF_O[1:], EF_T[1:], msg="OpenMM and TINKER forces are different", delta=0.01)

    def test_energy_rmsd(self):
        """ Compare OpenMM vs. TINKER optimized geometries with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER optimized geometry with AMOEBA")
        os.chdir("temp")
        tinkerpath=which('testgrad')
        if (which('testgrad') == ''):
            self.skipTest("TINKER programs are not in the PATH.")
        O = OpenMM(coords="hex.arc", pdb="prism.pdb", ffxml="amoebawater.xml", precision="double", \
               mmopts={'rigidWater':False, 'mutualInducedTargetEpsilon':1e-6})
        T = TINKER(coords="hex.arc", tinker_key="prism.key", tinkerpath=tinkerpath)
        EO, RO = O.energy_rmsd()
        ET, RT = T.energy_rmsd()
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same minimized energy to within 0.0001 kcal\n")
        self.assertAlmostEqual(EO, ET, msg="OpenMM and TINKER minimized energies are different", delta=0.0001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same RMSD to starting structure\n")
        self.assertAlmostEqual(RO, RT, msg="OpenMM and TINKER structures are different", delta=0.001)

    def test_interaction_energy(self):
        """ Compare OpenMM vs. TINKER interaction energies with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER interaction energy with AMOEBA")
        os.chdir("temp")
        tinkerpath=which('testgrad')
        if (which('testgrad') == ''):
            self.skipTest("TINKER programs are not in the PATH.")
        O = OpenMM(coords="hex.arc", pdb="prism.pdb", ffxml="amoebawater.xml", precision="double", \
               mmopts={'rigidWater':False, 'mutualInducedTargetEpsilon':1e-6})
        T = TINKER(coords="hex.arc", tinker_key="prism.key", tinkerpath=tinkerpath)
        IO = O.interaction_energy(fraga=range(9), fragb=range(9, 18))
        IT = T.interaction_energy(fraga=range(9), fragb=range(9, 18))
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same interaction energy\n")
        self.assertAlmostEqual(IO, IT, msg="OpenMM and TINKER interaction energies are different", delta=0.0001)

    def test_multipole_moments(self):
        """ Compare OpenMM vs. TINKER multipole moments with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER multipole moments with AMOEBA")
        os.chdir("temp")
        tinkerpath=which('testgrad')
        if (which('testgrad') == ''):
            self.skipTest("TINKER programs are not in the PATH.")
        O = OpenMM(coords="hex.arc", pdb="prism.pdb", ffxml="amoebawater.xml", precision="double", \
               mmopts={'rigidWater':False, 'mutualInducedTargetEpsilon':1e-6})
        T = TINKER(coords="hex.arc", tinker_key="prism.key", tinkerpath=tinkerpath)
        MO = O.multipole_moments(optimize=False)
        DO = np.array(MO['dipole'].values())
        QO = np.array(MO['quadrupole'].values())
        MT = T.multipole_moments(optimize=False)
        DT = np.array(MT['dipole'].values())
        QT = np.array(MT['quadrupole'].values())
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same dipole\n")
        self.assertNdArrayEqual(DO, DT, msg="OpenMM and TINKER dipoles are different", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same quadrupole\n")
        self.assertNdArrayEqual(QO, QT, msg="OpenMM and TINKER quadrupoles are different", delta=0.001)

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestOpenMM_vs_TINKER,self).shortDescription() + " (TINKER and OpenMM Engines)"

if __name__ == '__main__':           
    unittest.main()
