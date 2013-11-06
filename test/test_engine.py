import unittest
import sys, os, re
import forcebalance
import abc
import numpy as np
from __init__ import ForceBalanceTestCase
from forcebalance.nifty import *
from forcebalance.gmxio import GMX
from forcebalance.tinkerio import TINKER
from forcebalance.openmmio import OpenMM
from collections import OrderedDict

class TestAmber99SB(ForceBalanceTestCase):

    """ Amber99SB unit test consisting of ten structures of
    ACE-ALA-NME interacting with ACE-GLU-NME.  The tests check for
    whether the OpenMM, GMX, and TINKER Engines produce consistent
    results for:

    1) Single-point energies and forces over all ten structures
    2) Minimized energies and RMSD from the initial geometry for a selected structure
    3) Interaction energies between the two molecules over all ten structures
    4) Multipole moments of a selected structure
    5) Multipole moments of a selected structure after geometry optimization
    6) Normal modes of a selected structure
    7) Normal modes of a selected structure after geometry optimization

    If the engines are setting up the calculation correctly, then the
    remaining differences between results are due to differences in
    the parameter files or software implementations.

    The criteria in this unit test are more stringent than normal
    simulations.  In order for the software packages to agree to
    within the criteria, I had to do the following:

    - Remove improper dihedrals from the force field, because there is
    an ambiguity in the atom ordering which leads to force differences
    - Increase the number of decimal points in the "fudgeQQ" parameter
    in the GROMACS .itp file
    - Change the "electric" conversion factor in the TINKER .prm file
    - Compile GROMACS in double precision

    Residual errors are as follows:
    Potential energies: <0.01 kJ/mol (<1e-4 fractional error)
    Forces: <0.1 kJ/mol/nm (<1e-3 fractional error)
    Energy of optimized geometry: < 0.001 kcal/mol
    RMSD from starting structure: < 0.001 Angstrom
    Interaction energies: < 0.0001 kcal/mol
    Multipole moments: < 0.001 Debye / Debye Angstrom
    Multipole moments (optimized): < 0.01 Debye / Debye Angstrom
    Vibrational frequencies: < 0.5 wavenumber (~ 1e-4 fractional error)
    Vibrational eigenvectors: < 0.01
    """

    def setUp(self):
        tinkerpath=which('testgrad')
        gmxsuffix='_d'
        gmxpath=which('mdrun'+gmxsuffix)
        self.logger.debug("\nBuilding options for target...\n")
        self.cwd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), "test", "files", "amber_alaglu"))
        if not os.path.exists("temp"): os.makedirs("temp")
        os.chdir("temp")
        for i in ["topol.top", "shot.mdp", "a99sb.xml", "a99sb.prm", "all.gro", "all.arc", "AceGluNme.itp", "AceAlaNme.itp", "a99sb.itp"]:
            os.system("ln -fs ../%s" % i)
        self.engines = OrderedDict()
        # Set up GMX engine
        if gmxpath != '':
            self.engines['GMX'] = GMX(coords="all.gro", gmx_top="topol.top", gmx_mdp="shot.mdp", gmxpath=gmxpath, gmxsuffix=gmxsuffix)
        else: logger.warn("GROMACS cannot be found, skipping GMX tests.")
        # Set up TINKER engine
        if tinkerpath != '':
            self.engines['TINKER'] = TINKER(coords="all.arc", tinker_key="alaglu.key", tinkerpath=tinkerpath)
        else: logger.warn("TINKER cannot be found, skipping TINKER tests.")
        # Set up OpenMM engine
        openmm = False
        try:
            import simtk.openmm 
            openmm = True
        except: logger.warn("OpenMM cannot be imported, skipping OpenMM tests.")
        if openmm: self.engines['OpenMM'] = OpenMM(coords="all.gro", pdb="conf.pdb", ffxml="a99sb.xml", platname="CUDA", precision="double")
        self.addCleanup(os.system, 'cd .. ; rm -rf temp')

    def test_energy_force(self):
        """ Compare GMX, OpenMM, and TINKER energy and forces using AMBER force field """
        if len(self.engines) < 2:
            self.skipTest("Don't have two engines to compare")
        printcool("Compare GMX, OpenMM, and TINKER energy and forces using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_force()
        for i, n1 in enumerate(self.engines.keys()):
            for n2 in self.engines.keys()[:i]:
                self.assertNdArrayEqual(Data[n1][:,0], Data[n2][:,0], delta=0.01, msg="%s and %s energies are different" % (n1, n2))
                self.assertNdArrayEqual(Data[n1][:,1:].flatten(), Data[n2][:,1:].flatten(), \
                                            delta=0.1, msg="%s and %s forces are different" % (n1, n2))

    def test_optimized_geometries(self):
        """ Compare GMX, OpenMM, and TINKER optimized geometries and RMSD using AMBER force field """
        if len(self.engines) < 2:
            self.skipTest("Don't have two engines to compare")
        printcool("Compare GMX, OpenMM, and TINKER optimized geometries and RMSD using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_rmsd(5)
        for i, n1 in enumerate(self.engines.keys()):
            for n2 in self.engines.keys()[:i]:
                self.assertAlmostEqual(Data[n1][0], Data[n2][0], delta=0.001, \
                                           msg="%s and %s optimized energies are different" % (n1, n2))
                self.assertAlmostEqual(Data[n1][1], Data[n2][1], delta=0.001, \
                                           msg="%s and %s RMSD from starting structure are different" % (n1, n2))
                
    def test_interaction_energies(self):
        """ Compare GMX, OpenMM, and TINKER interaction energies using AMBER force field """
        if len(self.engines) < 2:
            self.skipTest("Don't have two engines to compare")
        printcool("Compare GMX, OpenMM, and TINKER interaction energies using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.interaction_energy(fraga=range(22), fragb=range(22, 49))
        for i, n1 in enumerate(self.engines.keys()):
            for n2 in self.engines.keys()[:i]:
                self.assertNdArrayEqual(Data[n1], Data[n2], delta=0.0001, \
                                           msg="%s and %s interaction energies are different" % (n1, n2))
        
    def test_multipole_moments(self):
        """ Compare GMX, OpenMM, and TINKER multipole moments using AMBER force field """
        if len(self.engines) < 2:
            self.skipTest("Don't have two engines to compare")
        printcool("Compare GMX, OpenMM, and TINKER multipole moments using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=False)
        for i, n1 in enumerate(self.engines.keys()):
            for n2 in self.engines.keys()[:i]:
                d1 = np.array(Data[n1]['dipole'].values())
                d2 = np.array(Data[n2]['dipole'].values())
                q1 = np.array(Data[n1]['quadrupole'].values())
                q2 = np.array(Data[n2]['quadrupole'].values())
                self.assertNdArrayEqual(d1, d2, delta=0.001, msg="%s and %s dipole moments are different" % (n1, n2))
                self.assertNdArrayEqual(q1, q2, delta=0.001, msg="%s and %s quadrupole moments are different" % (n1, n2))

    def test_multipole_moments_optimized(self):
        """ Compare GMX, OpenMM, and TINKER multipole moments at optimized geometries """
        #==================================================#
        #| Geometry-optimized multipole moments; requires |#
        #| double precision in order to pass!             |#
        #==================================================#
        if len(self.engines) < 2:
            self.skipTest("Don't have two engines to compare")
        printcool("Compare GMX, OpenMM, and TINKER multipole moments at optimized geometries")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=True)
        for i, n1 in enumerate(self.engines.keys()):
            for n2 in self.engines.keys()[:i]:
                d1 = np.array(Data[n1]['dipole'].values())
                d2 = np.array(Data[n2]['dipole'].values())
                q1 = np.array(Data[n1]['quadrupole'].values())
                q2 = np.array(Data[n2]['quadrupole'].values())
                self.assertNdArrayEqual(d1, d2, delta=0.02, msg="%s and %s dipole moments are different at optimized geometry" % (n1, n2))
                self.assertNdArrayEqual(q1, q2, delta=0.02, msg="%s and %s quadrupole moments are different at optimized geometry" % (n1, n2))
        
    def test_normal_modes(self):
        """ Compare GMX and TINKER normal modes """
        if 'TINKER' not in self.engines or 'GMX' not in self.engines:
            self.skipTest("Need TINKER and GMX engines to compare")
        printcool("Compare GMX and TINKER normal modes")
        FreqG, ModeG = self.engines['GMX'].normal_modes(shot=5, optimize=False)
        FreqT, ModeT = self.engines['TINKER'].normal_modes(shot=5, optimize=False)
        for vg, vt, mg, mt in zip(FreqG, FreqT, ModeG, ModeT):
            if vt < 0: continue
            # Wavefunction tolerance is half a wavenumber.
            self.assertAlmostEqual(vg, vt, delta=0.5, msg="GMX and TINKER vibrational frequencies are different")
            for a in range(len(mg)):
                try:
                    self.assertNdArrayEqual(mg[a], mt[a], delta=0.01, msg="GMX and TINKER normal modes are different")
                except:
                    self.assertNdArrayEqual(mg[a], -1.0*mt[a], delta=0.01, msg="GMX and TINKER normal modes are different")

    def test_normal_modes_optimized(self):
        """ Compare GMX and TINKER normal modes at optimized geometry """
        if 'TINKER' not in self.engines or 'GMX' not in self.engines:
            self.skipTest("Need TINKER and GMX engines to compare")
        printcool("Compare GMX and TINKER normal modes at optimized geometry")
        FreqG, ModeG = self.engines['GMX'].normal_modes(shot=5, optimize=True)
        FreqT, ModeT = self.engines['TINKER'].normal_modes(shot=5, optimize=True)
        for vg, vt, mg, mt in zip(FreqG, FreqT, ModeG, ModeT):
            self.assertAlmostEqual(vg, vt, delta=0.5, msg="GMX and TINKER vibrational frequencies are different at optimized geometry")
            for a in range(len(mg)):
                try:
                    self.assertNdArrayEqual(mg[a], mt[a], delta=0.01, msg="GMX and TINKER normal modes are different at optimized geometry")
                except:
                    self.assertNdArrayEqual(mg[a], -1.0*mt[a], delta=0.01, msg="GMX and TINKER normal modes are different at optimized geometry")
            


class TestAmoebaWater6(ForceBalanceTestCase):

    """ AMOEBA unit test consisting of a water hexamer.  The test
    checks for whether the OpenMM and TINKER Engines produce
    consistent results for:

    1) Single-point energy and force
    2) Minimized energies and RMSD from the initial geometry
    3) Interaction energies between two groups of molecules
    4) Multipole moments
    5) Multipole moments after geometry optimization

    Due to careful validation of OpenMM, the results agree with TINKER
    to within very stringent criteria.  Residual errors are as follows:

    Potential energies: <0.001 kJ/mol (<1e-5 fractional error)
    Forces: <0.01 kJ/mol/nm (<1e-4 fractional error)
    Energy of optimized geometry: < 0.0001 kcal/mol
    RMSD from starting structure: < 0.001 Angstrom
    Interaction energies: < 0.0001 kcal/mol
    Multipole moments: < 0.001 Debye / Debye Angstrom
    Multipole moments (optimized): < 0.01 Debye / Debye Angstrom
    """

    def setUp(self):
        self.logger.debug("\nBuilding options for target...\n")
        self.cwd = os.getcwd()
        os.chdir(os.path.join(os.getcwd(), "test", "files", "amoeba_h2o6"))
        if not os.path.exists("temp"): os.makedirs("temp")
        os.chdir("temp")
        os.system("ln -s ../prism.pdb")
        os.system("ln -s ../prism.key")
        os.system("ln -s ../hex.arc")
        os.system("ln -s ../water.prm")
        os.system("ln -s ../amoebawater.xml")
        self.O = OpenMM(coords="hex.arc", pdb="prism.pdb", ffxml="amoebawater.xml", precision="double", \
                            mmopts={'rigidWater':False, 'mutualInducedTargetEpsilon':1e-6})
        tinkerpath=which('testgrad')
        if (which('testgrad') != ''):
            self.T = TINKER(coords="hex.arc", tinker_key="prism.key", tinkerpath=tinkerpath)
        os.chdir("..")
        self.addCleanup(os.system, 'rm -rf temp')

    def test_energy_force(self):
        """ Compare OpenMM vs. TINKER energy and forces with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER energy and force with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        EF_O = self.O.energy_force()[0]
        EF_T = self.T.energy_force()[0]
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same AMOEBA energy to within 0.001 kJ\n")
        self.assertAlmostEqual(EF_O[0], EF_T[0], msg="OpenMM and TINKER energies are different", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Forces give the same AMOEBA energy to within 0.01 kJ/mol/nm\n")
        self.assertNdArrayEqual(EF_O[1:], EF_T[1:], msg="OpenMM and TINKER forces are different", delta=0.01)

    def test_energy_rmsd(self):
        """ Compare OpenMM vs. TINKER optimized geometries with AMOEBA force field """
        self.skipTest("Need to reduce dependence on the TINKER build")
        printcool("Testing OpenMM vs. TINKER optimized geometry with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        EO, RO = self.O.energy_rmsd()
        ET, RT = self.T.energy_rmsd()
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same minimized energy to within 0.0001 kcal\n")
        self.assertAlmostEqual(EO, ET, msg="OpenMM and TINKER minimized energies are different", delta=0.0001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same RMSD to starting structure\n")
        self.assertAlmostEqual(RO, RT, msg="OpenMM and TINKER structures are different", delta=0.001)

    def test_interaction_energy(self):
        """ Compare OpenMM vs. TINKER interaction energies with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER interaction energy with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        IO = self.O.interaction_energy(fraga=range(9), fragb=range(9, 18))
        IT = self.T.interaction_energy(fraga=range(9), fragb=range(9, 18))
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same interaction energy\n")
        self.assertAlmostEqual(IO, IT, msg="OpenMM and TINKER interaction energies are different", delta=0.0001)

    def test_multipole_moments(self):
        """ Compare OpenMM vs. TINKER multipole moments with AMOEBA force field """
        printcool("Testing OpenMM vs. TINKER multipole moments with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        MO = self.O.multipole_moments(optimize=False)
        DO = np.array(MO['dipole'].values())
        QO = np.array(MO['quadrupole'].values())
        MT = self.T.multipole_moments(optimize=False)
        DT = np.array(MT['dipole'].values())
        QT = np.array(MT['quadrupole'].values())
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same dipole\n")
        self.assertNdArrayEqual(DO, DT, msg="OpenMM and TINKER dipoles are different", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same quadrupole\n")
        self.assertNdArrayEqual(QO, QT, msg="OpenMM and TINKER quadrupoles are different", delta=0.001)

    def test_multipole_moments_optimized(self):
        """ Compare OpenMM vs. TINKER multipole moments with AMOEBA force field """
        self.skipTest("Need to reduce dependence on the TINKER build")
        printcool("Testing OpenMM vs. TINKER multipole moments with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        MO1 = self.O.multipole_moments(optimize=True)
        DO1 = np.array(MO1['dipole'].values())
        QO1 = np.array(MO1['quadrupole'].values())
        MT1 = self.T.multipole_moments(optimize=True)
        DT1 = np.array(MT1['dipole'].values())
        QT1 = np.array(MT1['quadrupole'].values())
        os.chdir("..")
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same dipole when geometries are optimized\n")
        self.assertNdArrayEqual(DO1, DT1, msg="OpenMM and TINKER dipoles are different when geometries are optimized", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the same quadrupole when geometries are optimized\n")
        self.assertNdArrayEqual(QO1, QT1, msg="OpenMM and TINKER quadrupoles are different when geometries are optimized", delta=0.01)

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestAmoebaWater6,self).shortDescription() + " (TINKER and OpenMM Engines)"

if __name__ == '__main__':           
    unittest.main()
