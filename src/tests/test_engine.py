from __future__ import absolute_import
from builtins import zip
from builtins import range
import pytest
from forcebalance.nifty import *
from forcebalance.gmxio import GMX
from forcebalance.tinkerio import TINKER
from forcebalance.openmmio import OpenMM
from collections import OrderedDict
from .__init__ import ForceBalanceTestCase, check_for_openmm

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
    - Increase two torsional barriers to ensure optimizer converges
    to the same local minimum consistently
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
    Vibrational eigenvectors: < 0.05 (on 11/2019, updated these)
    """

    @classmethod
    def setup_class(cls):
        """
        setup any state specific to the execution of the given class (which usually contains tests).
        """
        super(TestAmber99SB, cls).setup_class()
        tinkerpath = which('testgrad')
        # try to find mdrun_d or gmx_d
        # gmx should be built with config -DGMX_DOUBLE=ON
        gmxpath = which('mdrun_d') or which('gmx_d')
        gmxsuffix = '_d'
        # Tests will FAIL if use single precision gromacs
        # gmxpath = which('mdrun') or which('gmx')
        # gmxsuffix = ''
        # self.logger.debug("\nBuilding options for target...\n")
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, "files", "amber_alaglu"))
        cls.tmpfolder = os.path.join(cls.cwd, "files", "amber_alaglu", "temp")
        if not os.path.exists(cls.tmpfolder):
            os.makedirs(cls.tmpfolder)
        os.chdir(cls.tmpfolder)
        for i in ["topol.top", "shot.mdp", "a99sb.xml", "a99sb.prm", "all.gro", "all.arc", "AceGluNme.itp", "AceAlaNme.itp", "a99sb.itp"]:
            os.system("ln -fs ../%s" % i)
        cls.engines = OrderedDict()
        # Set up GMX engine
        if gmxpath != '':
            cls.engines['GMX'] = GMX(coords="all.gro", gmx_top="topol.top", gmx_mdp="shot.mdp", gmxpath=gmxpath, gmxsuffix=gmxsuffix)
        else:
            logger.warning("GROMACS cannot be found, skipping GMX tests.")
        # Set up TINKER engine
        if tinkerpath != '':
            cls.engines['TINKER'] = TINKER(coords="all.arc", tinker_key="alaglu.key", tinkerpath=tinkerpath)
        else:
            logger.warning("TINKER cannot be found, skipping TINKER tests.")
        # Set up OpenMM engine
        try:
            try:
                import openmm
            except ImportError:
                import simtk.openmm
            cls.engines['OpenMM'] = OpenMM(coords="all.gro", pdb="conf.pdb", ffxml="a99sb.xml", platname="Reference", precision="double")
        except:
            logger.warning("OpenMM cannot be imported, skipping OpenMM tests.")

    @classmethod
    def teardown_class(cls):
        """
        teardown any state that was previously setup with a call to setup_class.
        """
        os.chdir(cls.cwd)
        # shutil.rmtree(cls.cwd, "files", "amber_alaglu", "temp")

    def setup_method(self):
        os.chdir(self.tmpfolder)
    
    def test_energy_force(self):
        """ Test GMX, OpenMM, and TINKER energy and forces using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER energy and forces using AMBER force field")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_force()
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_energy_force.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[list(self.engines.keys())[0]])
        fin = os.path.join(datadir, 'test_energy_force.dat')
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            np.testing.assert_allclose(Data[n1][:,0], RefData[:,0], rtol=0, atol=0.01,
                                    err_msg="%s energies do not match the reference" % (n1))
            np.testing.assert_allclose(Data[n1][:,1:].flatten(), RefData[:,1:].flatten(),
                                    rtol=0, atol=0.1, err_msg="%s forces do not match the reference" % (n1))

    def test_optimized_geometries(self):
        """ Test GMX, OpenMM, and TINKER optimized geometries and RMSD using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER optimized geometries and RMSD using AMBER force field")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_rmsd(5)
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_optimized_geometries.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[list(self.engines.keys())[0]])
        fin = os.path.join(datadir, 'test_optimized_geometries.dat')
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            print("%s vs Reference energies:" % n1, Data[n1][0], RefData[0])
        for n1 in self.engines.keys():
            np.testing.assert_allclose(Data[n1][0], RefData[0], rtol=0, atol=0.001,
                                   err_msg="%s optimized energies do not match the reference" % n1)
            np.testing.assert_allclose(Data[n1][1], RefData[1], rtol=0, atol=0.001,
                                   err_msg="%s RMSD from starting structure do not match the reference" % n1)

    def test_interaction_energies(self):
        """ Test GMX, OpenMM, and TINKER interaction energies using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER interaction energies using AMBER force field")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.interaction_energy(fraga=list(range(22)), fragb=list(range(22, 49)))
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_interaction_energies.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[list(self.engines.keys())[0]])
        fin = os.path.join(datadir, 'test_interaction_energies.dat')
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            np.testing.assert_allclose(Data[n1], RefData, rtol=0, atol=0.0001,
                                    err_msg="%s interaction energies do not match the reference" % n1)

    def test_multipole_moments(self):
        """ Test GMX, OpenMM, and TINKER multipole moments using AMBER force field """
        printcool("Test GMX, OpenMM, and TINKER multipole moments using AMBER force field")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=False)
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array(list(Data[list(self.engines.keys())[0]]['dipole'].values())))
            fout = os.path.join(datadir, 'test_multipole_moments.quadrupole.dat')
            np.savetxt(fout, np.array(list(Data[list(self.engines.keys())[0]]['quadrupole'].values())))
        RefDip = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.dipole.dat'))
        RefQuad = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.quadrupole.dat'))
        for n1 in self.engines.keys():
            d1 = np.array(list(Data[n1]['dipole'].values()))
            q1 = np.array(list(Data[n1]['quadrupole'].values()))
            np.testing.assert_allclose(d1, RefDip, rtol=0, atol=0.001, err_msg="%s dipole moments do not match the reference" % n1)
            np.testing.assert_allclose(q1, RefQuad, rtol=0, atol=0.001, err_msg="%s quadrupole moments do not match the reference" % n1)

    def test_multipole_moments_optimized(self):
        """ Test GMX, OpenMM, and TINKER multipole moments at optimized geometries """
        #==================================================#
        #| Geometry-optimized multipole moments; requires |#
        #| double precision in order to pass!             |#
        #==================================================#
        printcool("Test GMX, OpenMM, and TINKER multipole moments at optimized geometries")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=True)
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array(list(Data[list(self.engines.keys())[0]]['dipole'].values())))
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat')
            np.savetxt(fout, np.array(list(Data[list(self.engines.keys())[0]]['quadrupole'].values())))
        RefDip = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat'))
        RefQuad = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat'))
        for n1 in self.engines.keys():
            d1 = np.array(list(Data[n1]['dipole'].values()))
            q1 = np.array(list(Data[n1]['quadrupole'].values()))
            np.testing.assert_allclose(d1, RefDip, rtol=0, atol=0.02, err_msg="%s dipole moments at optimized geometry do not match the reference" % n1)
            np.testing.assert_allclose(q1, RefQuad, rtol=0, atol=0.02, err_msg="%s quadrupole moments at optimized geometry do not match the reference" % n1)

    def test_normal_modes(self):
        """ Test GMX TINKER and OpenMM normal modes """
        printcool("Test GMX, TINKER, OpenMM normal modes")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        FreqG, ModeG = self.engines['GMX'].normal_modes(shot=5, optimize=False)
        FreqT, ModeT = self.engines['TINKER'].normal_modes(shot=5, optimize=False)
        FreqO, ModeO = self.engines['OpenMM'].normal_modes(shot=5, optimize=False)
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_normal_modes.freq.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, FreqT)
            fout = os.path.join(datadir, 'test_normal_modes.mode.dat.npy')
            # Need to save as binary data since it's a multidimensional array
            np.save(fout, ModeT)
        FreqRef = np.loadtxt(os.path.join(datadir, 'test_normal_modes.freq.dat'))
        ModeRef = np.load(os.path.join(datadir, 'test_normal_modes.mode.dat.npy'))
        for Freq, Mode, Name in [(FreqG, ModeG, 'GMX'), (FreqT, ModeT, 'TINKER'), (FreqO, ModeO, 'OpenMM')]:
            iv = -1
            for v, vr, m, mr in zip(Freq, FreqRef, Mode, ModeRef):
                iv += 1
                # Count vibrational modes. Stochastic issue seems to occur for a mode within the lowest 3.
                if vr < 0: continue# or iv < 3: continue
                # Frequency tolerance is half a wavenumber.
                np.testing.assert_allclose(v, vr, rtol=0, atol=0.5,
                                           err_msg="%s vibrational frequencies do not match the reference" % Name)
                delta = 0.05
                for a in range(len(m)):
                    try:
                        np.testing.assert_allclose(m[a], mr[a], rtol=0, atol=delta,
                                                   err_msg="%s normal modes do not match the reference" % Name)
                    except:
                        np.testing.assert_allclose(m[a], -1.0*mr[a], rtol=0, atol=delta,
                                                   err_msg="%s normal modes do not match the reference" % Name)

    def test_normal_modes_optimized(self):
        """ Test GMX TINKER and OpenMM normal modes at optimized geometry """
        printcool("Test GMX, TINKER, OpenMM normal modes at optimized geometry")
        missing_pkgs = []
        for eng in ['TINKER', 'GMX', 'OpenMM']:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ', '.join(missing_pkgs))
        FreqG, ModeG = self.engines['GMX'].normal_modes(shot=5, optimize=True)
        FreqT, ModeT = self.engines['TINKER'].normal_modes(shot=5, optimize=True)
        FreqO, ModeO = self.engines['OpenMM'].normal_modes(shot=5, optimize=True)
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_normal_modes_optimized.freq.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, FreqT)
            fout = os.path.join(datadir, 'test_normal_modes_optimized.mode.dat')
            # Need to save as binary data since it's a multidimensional array
            np.save(fout, ModeT)
        FreqRef = np.loadtxt(os.path.join(datadir, 'test_normal_modes_optimized.freq.dat'))
        ModeRef = np.load(os.path.join(datadir, 'test_normal_modes_optimized.mode.dat.npy'))
        for Freq, Mode, Name in [(FreqG, ModeG, 'GMX'), (FreqT, ModeT, 'TINKER'), (FreqO, ModeO, 'OpenMM')]:
            iv = -1
            for v, vr, m, mr in zip(Freq, FreqRef, Mode, ModeRef):
                iv += 1
                # Count vibrational modes. Stochastic issue seems to occur for a mode within the lowest 3.
                if vr < 0: continue# or iv < 3: continue
                # Frequency tolerance is half a wavenumber.
                np.testing.assert_allclose(v, vr, rtol=0, atol=0.5,
                                           err_msg="%s vibrational frequencies do not match the reference" % Name)
                delta = 0.05
                for a in range(len(m)):
                    try:
                        np.testing.assert_allclose(m[a], mr[a], rtol=0, atol=delta,
                                                   err_msg="%s normal modes do not match the reference" % Name)
                    except:
                        np.testing.assert_allclose(m[a], -1.0*mr[a], rtol=0, atol=delta,
                                                   err_msg="%s normal modes do not match the reference" % Name)


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
    @classmethod
    def setup_class(cls):
        if not check_for_openmm(): pytest.skip("No OpenMM modules found.")
        super(TestAmoebaWater6, cls).setup_class()
        #self.logger.debug("\nBuilding options for target...\n")
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, "files", "amoeba_h2o6"))
        cls.tmpfolder = os.path.join(cls.cwd, "files", "amoeba_h2o6", "temp")
        if not os.path.exists(cls.tmpfolder):
            os.makedirs(cls.tmpfolder)
        os.chdir(cls.tmpfolder)
        os.system("ln -s ../prism.pdb")
        os.system("ln -s ../prism.key")
        os.system("ln -s ../hex.arc")
        os.system("ln -s ../water.prm")
        os.system("ln -s ../amoebawater.xml")
        cls.O = OpenMM(coords="hex.arc", pdb="prism.pdb", ffxml="amoebawater.xml", precision="double", \
                            mmopts={'rigidWater':False, 'mutualInducedTargetEpsilon':1e-6})
        tinkerpath = which('testgrad')
        if tinkerpath:
            cls.T = TINKER(coords="hex.arc", tinker_key="prism.key", tinkerpath=tinkerpath)

    @classmethod
    def teardown_class(cls):
        """
        teardown any state that was previously setup with a call to setup_class.
        """
        os.chdir(cls.cwd)
        # shutil.rmtree(cls.cwd, "files", "amoeba_h2o6", "temp")
        
    def setup_method(self):
        os.chdir(self.tmpfolder)

    def test_energy_force(self):
        """ Test OpenMM and TINKER energy and forces with AMOEBA force field """
        printcool("Testing OpenMM and TINKER energy and force with AMOEBA")
        if not hasattr(self, 'T'):
            pytest.skip("TINKER programs are not in the PATH.")
        EF_O = self.O.energy_force()[0]
        EF_T = self.T.energy_force()[0]
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_energy_force.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, EF_T)
        EF_R = np.loadtxt(os.path.join(datadir, 'test_energy_force.dat'))
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct AMOEBA energy to within 0.001 kJ\n")
        np.testing.assert_allclose(EF_O[0], EF_R[0],
                                   err_msg="OpenMM energy does not match the reference", rtol=0, atol=0.001)
        np.testing.assert_allclose(EF_T[0], EF_R[0],
                                   err_msg="TINKER energy does not match the reference", rtol=0, atol=0.001)
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct AMOEBA force to within 0.01 kJ/mol/nm\n")
        np.testing.assert_allclose(EF_O[1:], EF_R[1:],
                                   err_msg="OpenMM forces do not match the reference", rtol=0, atol=0.01)
        np.testing.assert_allclose(EF_T[1:], EF_R[1:],
                                   err_msg="TINKER forces do not match the reference", rtol=0, atol=0.01)

    def test_energy_rmsd(self):
        """ Test OpenMM and TINKER optimized geometries with AMOEBA force field """
        pytest.skip("Need to reduce dependence on the TINKER build")
        printcool("Testing OpenMM and TINKER optimized geometry with AMOEBA")
        if not hasattr(self, 'T'):
            pytest.skip("TINKER programs are not in the PATH.")
        EO, RO = self.O.energy_rmsd()
        ET, RT = self.T.energy_rmsd()
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_energy_rmsd.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array([ET, RT]))
        RefData = os.path.join(datadir, 'test_energy_rmsd.dat')
        ERef = RefData[0]
        RRef = RefData[1]
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct minimized energy to within 0.0001 kcal\n")
        np.testing.assert_allclose(EO, ERef,
                                   err_msg="OpenMM minimized energy does not match the reference", rtol=0, atol=0.0001)
        np.testing.assert_allclose(ET, ERef,
                                   err_msg="TINKER minimized energy does not match the reference", rtol=0, atol=0.0001)
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct RMSD to starting structure\n")
        np.testing.assert_allclose(RO, RRef,
                                   err_msg="OpenMM RMSD does not match the reference", rtol=0, atol=0.001)
        np.testing.assert_allclose(RT, RRef,
                                   err_msg="TINKER RMSD does not match the reference", rtol=0, atol=0.001)

    def test_interaction_energy(self):
        """ Test OpenMM and TINKER interaction energies with AMOEBA force field """
        printcool("Testing OpenMM and TINKER interaction energy with AMOEBA")
        if not hasattr(self, 'T'):
            pytest.skip("TINKER programs are not in the PATH.")
        IO = self.O.interaction_energy(fraga=list(range(9)), fragb=list(range(9, 18)))
        IT = self.T.interaction_energy(fraga=list(range(9)), fragb=list(range(9, 18)))
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_interaction_energy.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, np.array([IT]))
        IR = np.loadtxt(os.path.join(datadir, 'test_interaction_energy.dat'))
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct interaction energy\n")
        np.testing.assert_allclose(IO, IR,
                                   err_msg="OpenMM interaction energies do not match the reference", rtol=0, atol=0.0001)
        np.testing.assert_allclose(IT, IR,
                                   err_msg="TINKER interaction energies do not match the reference", rtol=0, atol=0.0001)

    def test_multipole_moments(self):
        """ Test OpenMM and TINKER multipole moments with AMOEBA force field """
        printcool("Testing OpenMM and TINKER multipole moments with AMOEBA")
        if not hasattr(self, 'T'):
            pytest.skip("TINKER programs are not in the PATH.")
        MO = self.O.multipole_moments(optimize=False)
        DO = np.array(list(MO['dipole'].values()))
        QO = np.array(list(MO['quadrupole'].values()))
        MT = self.T.multipole_moments(optimize=False)
        DT = np.array(list(MT['dipole'].values()))
        QT = np.array(list(MT['quadrupole'].values()))
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, DT)
            fout = os.path.join(datadir, 'test_multipole_moments.quadrupole.dat')
            np.savetxt(fout, QT)
        DR = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.dipole.dat'))
        QR = np.loadtxt(os.path.join(datadir, 'test_multipole_moments.quadrupole.dat'))
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct dipole\n")
        np.testing.assert_allclose(DO, DR,
                                   err_msg="OpenMM dipoles do not match the reference", rtol=0, atol=0.001)
        np.testing.assert_allclose(DT, DR,
                                   err_msg="TINKER dipoles do not match the reference", rtol=0, atol=0.001)
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct quadrupole\n")
        np.testing.assert_allclose(QO, QR,
                                   err_msg="OpenMM quadrupoles do not match the reference", rtol=0, atol=0.001)
        np.testing.assert_allclose(QT, QR,
                                   err_msg="TINKER quadrupoles do not match the reference", rtol=0, atol=0.001)

    def test_multipole_moments_optimized(self):
        """ Test OpenMM and TINKER multipole moments with AMOEBA force field """
        pytest.skip("Need to reduce dependence on the TINKER build")
        printcool("Testing OpenMM and TINKER multipole moments with AMOEBA")
        if not hasattr(self, 'T'):
            pytest.skip("TINKER programs are not in the PATH.")
        MO1 = self.O.multipole_moments(optimize=True)
        DO1 = np.array(list(MO1['dipole'].values()))
        QO1 = np.array(list(MO1['quadrupole'].values()))
        MT1 = self.T.multipole_moments(optimize=True)
        DT1 = np.array(list(MT1['dipole'].values()))
        QT1 = np.array(list(MT1['quadrupole'].values()))
        datadir = os.path.join(self.cwd, 'files', 'test_engine', self.__class__.__name__)
        if SAVEDATA:
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat')
            if not os.path.exists(os.path.dirname(fout)): os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, DT1)
            fout = os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat')
            np.savetxt(fout, QT1)
        DR1 = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.dipole.dat'))
        QR1 = np.loadtxt(os.path.join(datadir, 'test_multipole_moments_optimized.quadrupole.dat'))
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct dipole when geometries are optimized\n")
        np.testing.assert_allclose(DO1, DR1, rtol=0, atol=0.001,
                                   err_msg="OpenMM dipoles do not match the reference when geometries are optimized")
        np.testing.assert_allclose(DT1, DR1, rtol=0, atol=0.001,
                                   err_msg="TINKER dipoles do not match the reference when geometries are optimized")
        #self.logger.debug(">ASSERT OpenMM and TINKER Engines give the correct quadrupole when geometries are optimized\n")
        np.testing.assert_allclose(QO1, QR1, rtol=0, atol=0.01,
                                   err_msg="OpenMM quadrupoles do not match the reference when geometries are optimized")
        np.testing.assert_allclose(QT1, QR1, rtol=0, atol=0.01,
                                   err_msg="TINKER quadrupoles do not match the reference when geometries are optimized")

