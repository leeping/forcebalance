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

# Set SAVEDATA to True and run the tests in order to save data
# to a file for future reference. This is easier to use for troubleshooting
# vs. comparing multiple programs against each other, b/c we don't know
# which one changed.
SAVEDATA=False

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
        if openmm: self.engines['OpenMM'] = OpenMM(coords="all.gro", pdb="conf.pdb", ffxml="a99sb.xml", platname="Reference", precision="double")
        self.addCleanup(os.system, 'cd .. ; rm -rf temp')

    def test_energy_force(self):
        """ Test GMX, OpenMM, and TINKER energy and forces using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER energy and forces using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_force()
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_energy_force.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[self.engines.keys()[0]])
        fin = os.path.join(datadir, 'test_energy_force.dat')
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            self.assertNdArrayEqual(Data[n1][:,0], RefData[:,0], delta=0.01, 
                                    msg="%s energies do not match the reference" % (n1))
            self.assertNdArrayEqual(Data[n1][:,1:].flatten(), RefData[:,1:].flatten(), 
                                    delta=0.1, msg="%s forces do not match the reference" % (n1))

    def test_optimized_geometries(self):
        """ Test GMX, OpenMM, and TINKER optimized geometries and RMSD using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER optimized geometries and RMSD using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_rmsd(5)
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_optimized_geometries.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[self.engines.keys()[0]])
        fin = os.path.join(datadir, 'test_optimized_geometries.dat')
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            self.assertAlmostEqual(Data[n1][0], RefData[0], delta=0.001,
                                   msg="%s optimized energies do not match the reference" % n1)
            self.assertAlmostEqual(Data[n1][1], RefData[1], delta=0.001,
                                   msg="%s RMSD from starting structure do not match the reference" % n1)
                
    def test_interaction_energies(self):
        """ Test GMX, OpenMM, and TINKER interaction energies using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER interaction energies using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.interaction_energy(fraga=range(22), fragb=range(22, 49))
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_interaction_energies.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[self.engines.keys()[0]])
        fin = os.path.join(datadir, 'test_interaction_energies.dat')
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            self.assertNdArrayEqual(Data[n1], RefData, delta=0.0001,
                                    msg="%s interaction energies do not match the reference" % n1)
        
    def test_multipole_moments(self):
        """ Test GMX, OpenMM, and TINKER multipole moments using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER multipole moments using AMBER force field")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=False)
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array(Data[self.engines.keys()[0]]['dipole'].values()))
            fout = os.path.join(datadir, 'test_multipole_moments.quadrupole.dat')
            np.savetxt(fout, np.array(Data[self.engines.keys()[0]]['quadrupole'].values()))
        RefDip = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.dipole.dat'))
        RefQuad = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.quadrupole.dat'))
        for n1 in self.engines.keys():
            d1 = np.array(Data[n1]['dipole'].values())
            q1 = np.array(Data[n1]['quadrupole'].values())
            self.assertNdArrayEqual(d1, RefDip, delta=0.001, msg="%s dipole moments do not match the reference" % n1)
            self.assertNdArrayEqual(q1, RefQuad, delta=0.001, msg="%s quadrupole moments do not match the reference" % n1)

    def test_multipole_moments_optimized(self):
        """ Test GMX, OpenMM, and TINKER multipole moments at optimized geometries """
        #==================================================#
        #| Geometry-optimized multipole moments; requires |#
        #| double precision in order to pass!             |#
        #==================================================#
        printcool("Test GMX, OpenMM, and TINKER multipole moments at optimized geometries")
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=True)
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array(Data[self.engines.keys()[0]]['dipole'].values()))
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat')
            np.savetxt(fout, np.array(Data[self.engines.keys()[0]]['quadrupole'].values()))
        RefDip = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat'))
        RefQuad = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat'))
        for n1 in self.engines.keys():
            d1 = np.array(Data[n1]['dipole'].values())
            q1 = np.array(Data[n1]['quadrupole'].values())
            self.assertNdArrayEqual(d1, RefDip, delta=0.02, msg="%s dipole moments at optimized geometry do not match the reference" % n1)
            self.assertNdArrayEqual(q1, RefQuad, delta=0.02, msg="%s quadrupole moments at optimized geometry do not match the reference" % n1)
        
    def test_normal_modes(self):
        """ Test GMX and TINKER normal modes """
        if 'TINKER' not in self.engines or 'GMX' not in self.engines:
            self.skipTest("Need TINKER and GMX engines")
        printcool("Test GMX and TINKER normal modes")
        FreqG, ModeG = self.engines['GMX'].normal_modes(shot=5, optimize=False)
        FreqT, ModeT = self.engines['TINKER'].normal_modes(shot=5, optimize=False)
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_normal_modes.freq.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, FreqT)
            fout = os.path.join(datadir, 'test_normal_modes.mode.dat')
            # Need to save as binary data since it's a multidimensional array
            np.save(fout, ModeT)
        FreqRef = np.loadtxt(os.path.join(datadir, 'test_normal_modes.freq.dat'))
        ModeRef = np.load(os.path.join(datadir, 'test_normal_modes.mode.dat.npy'))
        for Freq, Mode, Name in [(FreqG, ModeG, 'GMX'), (FreqT, ModeT, 'TINKER')]:
            for v, vr, m, mr in zip(Freq, FreqRef, Mode, ModeRef):
                if vr < 0: continue
                # Frequency tolerance is half a wavenumber.
                self.assertAlmostEqual(v, vr, delta=0.5, msg="%s vibrational frequencies do not match the reference" % Name)
                for a in range(len(m)):
                    try:
                        self.assertNdArrayEqual(m[a], mr[a], delta=0.01, msg="%s normal modes do not match the reference" % Name)
                    except:
                        self.assertNdArrayEqual(m[a], -1.0*mr[a], delta=0.01, msg="%s normal modes do not match the reference" % Name)

    def test_normal_modes_optimized(self):
        """ Test GMX and TINKER normal modes at optimized geometry """
        if 'TINKER' not in self.engines or 'GMX' not in self.engines:
            self.skipTest("Need TINKER and GMX engines")
        printcool("Test GMX and TINKER normal modes at optimized geometry")
        FreqG, ModeG = self.engines['GMX'].normal_modes(shot=5, optimize=True)
        FreqT, ModeT = self.engines['TINKER'].normal_modes(shot=5, optimize=True)
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_normal_modes_optimized.freq.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, FreqT)
            fout = os.path.join(datadir, 'test_normal_modes_optimized.mode.dat')
            # Need to save as binary data since it's a multidimensional array
            np.save(fout, ModeT)
        FreqRef = np.loadtxt(os.path.join(datadir, 'test_normal_modes_optimized.freq.dat'))
        ModeRef = np.load(os.path.join(datadir, 'test_normal_modes_optimized.mode.dat.npy'))
        for Freq, Mode, Name in [(FreqG, ModeG, 'GMX'), (FreqT, ModeT, 'TINKER')]:
            for v, vr, m, mr in zip(Freq, FreqRef, Mode, ModeRef):
                if vr < 0: continue
                # Frequency tolerance is half a wavenumber.
                self.assertAlmostEqual(v, vr, delta=0.5, msg="%s vibrational frequencies at optimized geometry do not match the reference" % Name)
                for a in range(len(m)):
                    try:
                        self.assertNdArrayEqual(m[a], mr[a], delta=0.01, msg="%s normal modes at optimized geometry do not match the reference" % Name)
                    except:
                        self.assertNdArrayEqual(m[a], -1.0*mr[a], delta=0.01, msg="%s normal modes at optimized geometry do not match the reference" % Name)


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
        """ Test OpenMM and TINKER energy and forces with AMOEBA force field """
        printcool("Testing OpenMM and TINKER energy and force with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        EF_O = self.O.energy_force()[0]
        EF_T = self.T.energy_force()[0]
        os.chdir("..")
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_energy_force.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, EF_T)
        EF_R = np.loadtxt(os.path.join(datadir, 'test_energy_force.dat'))
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct AMOEBA energy to within 0.001 kJ\n")
        self.assertAlmostEqual(EF_O[0], EF_R[0], msg="OpenMM energy does not match the reference", delta=0.001)
        self.assertAlmostEqual(EF_T[0], EF_R[0], msg="TINKER energy does not match the reference", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct AMOEBA force to within 0.01 kJ/mol/nm\n")
        self.assertNdArrayEqual(EF_O[1:], EF_R[1:], msg="OpenMM forces do not match the reference", delta=0.01)
        self.assertNdArrayEqual(EF_T[1:], EF_R[1:], msg="TINKER forces do not match the reference", delta=0.01)

    def test_energy_rmsd(self):
        """ Test OpenMM and TINKER optimized geometries with AMOEBA force field """
        self.skipTest("Need to reduce dependence on the TINKER build")
        printcool("Testing OpenMM and TINKER optimized geometry with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        EO, RO = self.O.energy_rmsd()
        ET, RT = self.T.energy_rmsd()
        os.chdir("..")
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_energy_rmsd.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array([ET, RT]))
        RefData = os.path.join(datadir, 'test_energy_rmsd.dat')
        ERef = RefData[0]
        RRef = RefData[1]
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct minimized energy to within 0.0001 kcal\n")
        self.assertAlmostEqual(EO, ERef, msg="OpenMM minimized energy does not match the reference", delta=0.0001)
        self.assertAlmostEqual(ET, ERef, msg="TINKER minimized energy does not match the reference", delta=0.0001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct RMSD to starting structure\n")
        self.assertAlmostEqual(RO, RRef, msg="OpenMM RMSD does not match the reference", delta=0.001)
        self.assertAlmostEqual(RT, RRef, msg="TINKER RMSD does not match the reference", delta=0.001)

    def test_interaction_energy(self):
        """ Test OpenMM and TINKER interaction energies with AMOEBA force field """
        printcool("Testing OpenMM and TINKER interaction energy with AMOEBA")
        os.chdir("temp")
        if not hasattr(self, 'T'):
            self.skipTest("TINKER programs are not in the PATH.")
        IO = self.O.interaction_energy(fraga=range(9), fragb=range(9, 18))
        IT = self.T.interaction_energy(fraga=range(9), fragb=range(9, 18))
        os.chdir("..")
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_interaction_energy.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array([IT]))
        IR = np.loadtxt(os.path.join(datadir, 'test_interaction_energy.dat'))
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct interaction energy\n")
        self.assertAlmostEqual(IO, IR, msg="OpenMM interaction energies do not match the reference", delta=0.0001)
        self.assertAlmostEqual(IT, IR, msg="TINKER interaction energies do not match the reference", delta=0.0001)

    def test_multipole_moments(self):
        """ Test OpenMM and TINKER multipole moments with AMOEBA force field """
        printcool("Testing OpenMM and TINKER multipole moments with AMOEBA")
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
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, DT)
            fout = os.path.join(datadir, 'test_multipole_moments.quadrupole.dat')
            np.savetxt(fout, QT)
        DR = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.dipole.dat'))
        QR = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.quadrupole.dat'))
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct dipole\n")
        self.assertNdArrayEqual(DO, DR, msg="OpenMM dipoles do not match the reference", delta=0.001)
        self.assertNdArrayEqual(DT, DR, msg="TINKER dipoles do not match the reference", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct quadrupole\n")
        self.assertNdArrayEqual(QO, QR, msg="OpenMM quadrupoles do not match the reference", delta=0.001)
        self.assertNdArrayEqual(QT, QR, msg="TINKER quadrupoles do not match the reference", delta=0.001)

    def test_multipole_moments_optimized(self):
        """ Test OpenMM and TINKER multipole moments with AMOEBA force field """
        self.skipTest("Need to reduce dependence on the TINKER build")
        printcool("Testing OpenMM and TINKER multipole moments with AMOEBA")
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
        datadir = os.path.join(sys.path[0], 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, DT1)
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat')
            np.savetxt(fout, QT1)
        DR1 = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat'))
        QR1 = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat'))
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct dipole when geometries are optimized\n")
        self.assertNdArrayEqual(DO1, DR1, msg="OpenMM dipoles do not match the reference when geometries are optimized", delta=0.001)
        self.assertNdArrayEqual(DT1, DR1, msg="TINKER dipoles do not match the reference when geometries are optimized", delta=0.001)
        self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct quadrupole when geometries are optimized\n")
        self.assertNdArrayEqual(QO1, QR1, msg="OpenMM quadrupoles do not match the reference when geometries are optimized", delta=0.01)
        self.assertNdArrayEqual(QT1, QR1, msg="TINKER quadrupoles do not match the reference when geometries are optimized", delta=0.01)

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestAmoebaWater6,self).shortDescription() + " (TINKER and OpenMM Engines)"

if __name__ == '__main__':           
    unittest.main()
