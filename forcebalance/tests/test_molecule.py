from __future__ import absolute_import
from builtins import str
import pytest
import os
import forcebalance.molecule
from .__init__ import ForceBalanceTestCase
import numpy as np

class TestPDBMolecule(ForceBalanceTestCase):

    @classmethod
    def setup_class(cls):
        cls.source = 'dms_conf.pdb'
        super(TestPDBMolecule, cls).setup_class()

    def setup_method(self, method):
        super(TestPDBMolecule, self).setup_method(method)
        os.chdir('files')
        try: self.molecule = forcebalance.molecule.Molecule(self.source, build_topology=False)
        except IOError:
            self.skipTest("Input pdb file test/files/%s doesn't exist" % self.source)
        except:
            self.fail("\nUnable to open pdb file")

    def teardown_method(self):
        os.system('rm -rf {name}.xyz {name}.gro {name}.arc'.format(name=self.source[:-4]))
        super(TestPDBMolecule, self).teardown_method()

    def test_xyz_conversion(self):
        """Check molecule conversion from pdb to xyz format"""
        self.logger.debug("\nCreating xyz file from pdb... ")
        self.molecule.write(self.source[:-3] + 'xyz')
        self.logger.debug("done\nTrying to read generated xyz file... ")
        try:
            molecule1 = forcebalance.molecule.Molecule(self.source[:-3] + 'xyz', build_topology=False)
            self.logger.debug("ok\n")
        except:
            self.fail("\nConversion to xyz format creates unreadable file")

        self.logger.debug("Checking that conversion has not changed molecule spatial coordinates\n")
        assert (self.molecule.Data['xyzs'][0] == molecule1.Data['xyzs'][0]).all(), "Conversion from pdb to xyz yields "\
            "different xyz coordinates\npdb:\n%s\n\ngro:\n%s\n" %(str(self.molecule.Data['xyzs'][0]),
                                                                  str(molecule1.Data['xyzs'][0]))

    def test_measure_distances(self):
        """Check measure distance functions"""
        result = np.array(self.molecule.measure_distances(1701, 1706))
        EXPECTED_DISTANCE_RESULTS = np.array([1.80])
        np.testing.assert_allclose(EXPECTED_DISTANCE_RESULTS,result,atol=0.01,
                                   err_msg="\nMeasured distance has changed from previously calculated values.\n"
                                           "If this seems reasonable, update EXPECTED_DISTANCE_RESULTS in test_"
                                           "molecule.py with these values")

    def test_measure_angles(self):
        """Check measure angle functions"""
        result = np.array(self.molecule.measure_angles(1771, 1769, 1772))
        EXPECTED_ANGLE_RESULTS = np.array([114.11])
        np.testing.assert_allclose(EXPECTED_ANGLE_RESULTS,result,atol=0.01,
                                   err_msg="\nMeasured angle has changed from previously calculated values.\n"
                                           "If this seems reasonable, update EXPECTED_ANGLE_RESULTS in "
                                           "test_molecule.py with these values")

    def test_measure_dihedrals(self):
        """Check measure dihedral functions"""
        result = np.array(self.molecule.measure_dihedrals(1709, 1706, 1701, 1702))
        EXPECTED_DIHEDRAL_RESULTS = np.array([176.54])
        np.testing.assert_allclose(EXPECTED_DIHEDRAL_RESULTS,result,atol=0.01,
                                   err_msg="\nMeasured dihedral angle has changed from previously calculated values.\n"
                                           "If this seems reasonable, update EXPECTED_DIHEDRAL_RESULTS "
                                           "in test_molecule.py with these values")

    def test_gro_conversion(self):
        """Check molecule conversion from pdb to gro format"""
        self.logger.debug("\nCreating gro file from pdb... ")
        self.molecule.write(self.source[:-3] + 'gro')
        self.logger.debug("done\nTrying to read generated gro file... ")
        try:
            molecule1 = forcebalance.molecule.Molecule(self.source[:-3] + 'gro', build_topology=False)
            self.logger.debug("ok\n")
        except:
            self.fail("Conversion to gro format creates unreadable file")

        self.logger.debug("\nChecking that conversion has not changed number of residues\n")
        assert len(self.molecule.Data['resid']) == len(molecule1.Data['resid'])

        self.logger.debug("Checking that conversion has not changed molecule spatial coordinates\n")
        msg = "\nConversion from pdb to gro yields different xyz coordinates\npdb:\n%s\n\ngro:\n" \
              "%s\n" % (str(self.molecule.Data['xyzs'][0]), str(molecule1.Data['xyzs'][0]))
        np.testing.assert_allclose(self.molecule.Data['xyzs'][0],  molecule1.Data['xyzs'][0], rtol=0, atol=0.001,
                                   err_msg=msg)


    def test_arc_conversion(self):
        """Check molecule conversion from pdb to arc format"""
        self.logger.debug("\nCreating arc file from pdb... ")
        self.molecule.Data['tinkersuf']=['']*len(self.molecule.Data['resname'])  # suppress topology warning
        self.molecule.write(self.source[:-3] + 'arc')
        self.logger.debug("done\nTrying to read generated gro file... ")
        try:
            molecule1 = forcebalance.molecule.Molecule(self.source[:-3] + 'arc',build_topology=False)
            self.logger.debug("ok\n")
        except:
            self.fail("Conversion to arc (TINKER) format creates unreadable file")

        self.logger.debug("Checking that conversion has not changed molecule spatial coordinates\n")

        msg = "Conversion from pdb to arc (TINKER) yields different number of residues"
        assert len(self.molecule.Data['resid']) == len(molecule1.Data['resid']), msg

        msg = "\nConversion from pdb to arc yields different xyz coordinates" \
              "\npdb:\n%s\n\narc:\n%s\n" %(str(self.molecule.Data['xyzs'][0]),
                                           str(molecule1.Data['xyzs'][0]))
        np.testing.assert_allclose(self.molecule.Data['xyzs'][0], molecule1.Data['xyzs'][0],
                                   atol=0.000001, err_msg=msg)

    def test_pdb_topology_build(self):
        """Check reading pdb with build_topology=True"""
        self.logger.debug("\nTrying to read molecule with topology... ")
        try:
            molecule = forcebalance.molecule.Molecule(self.source, build_topology=True)
            self.logger.debug("\ndone\nChecking molecule has correct number of residues")
        except:
            self.fail("Failed to load pdb with build_topology=True")

        assert len(self.molecule.Data['resid']) == len(molecule.Data['resid']), "Topology build yields " \
                                                                                "different number of residues"

class TestLipidGRO(ForceBalanceTestCase):
    @classmethod
    def setup_class(cls):
        super(TestLipidGRO, cls).setup_class()
        cls.source = 'lipid.gro'

    def setup_method(self, method):
        super(TestLipidGRO, self).setup_method(method)
        os.chdir('files')
        try: self.molecule = forcebalance.molecule.Molecule(self.source, toppbc=True)
        except IOError:
            pytest.skip("Input pdb file test/files/%s doesn't exist" % self.source)
        except:
            pytest.fail("Unable to open gro file")
            
    def test_measure_dihedrals(self):
        """Check measure dihedral functions"""
        result = np.array(self.molecule.measure_dihedrals(1131, 1112, 1113, 1114))
        EXPECTED_DIHEDRAL_RESULTS = np.array([-157.223])
        msg = "\nMeasured dihedral angle has changed from previously calculated values.\n" \
              "If this seems reasonable, update EXPECTED_DIHEDRAL_RESULTS in test_molecule.py with these values"
        np.testing.assert_allclose(EXPECTED_DIHEDRAL_RESULTS, result, atol=0.001, err_msg=msg)


    def test_lipid_molecules(self):
        """Check for the correct number of molecules in a rectangular cell with broken molecules"""
        self.logger.debug("\nTrying to read lipid conformation... ")
        assert len(self.molecule.molecules) == 3783, "Incorrect number of molecules for lipid structure"

class TestWaterPDB(ForceBalanceTestCase):
    @classmethod
    def setup_class(cls):
        super(TestWaterPDB, cls).setup_class()
        cls.source = 'waterbox500.pdb'

    def setup_method(self, method):
        super(TestWaterPDB, self).setup_method(method)
        os.chdir('files')
        try: self.molecule = forcebalance.molecule.Molecule(self.source, toppbc=True)
        except IOError:
            pytest.skip("Input pdb file test/files/%s doesn't exist" % self.source)
        except:
            pytest.fail("Unable to open pdb file")

    def test_water_molecules(self):
        """Check for the correct number of molecules in a cubic water box"""
        self.logger.debug("\nTrying to read water conformation... ")
        assert len(self.molecule.molecules) == 500, "Incorrect number of molecules for water structure"

class TestAlaGRO(ForceBalanceTestCase):
    @classmethod
    def setup_class(cls):
        super(TestAlaGRO, cls).setup_class()
        cls.source = 'ala.gro'

    def setup_method(self, method):
        super(TestAlaGRO, self).setup_method(method)
        os.chdir('files')
        try: self.molecule = forcebalance.molecule.Molecule(self.source)
        except IOError:
            pytest.skip("Input gro file test/files/%s doesn't exist" % self.source)
        except:
            pytest.fail("Unable to open gro file")

    def test_ala_molecules(self):
        """Check for the correct number of bonds in a simple molecule"""
        self.logger.debug("\nTrying to read alanine dipeptide conformation... ")
        assert len(self.molecule.bonds) == 21, "Incorrect number of bonds for alanine dipeptide structure"


class TestGalbPNPMol2(ForceBalanceTestCase):
    @classmethod
    def setup_class(cls):
        super(TestGalbPNPMol2, cls).setup_class()
        cls.source = 'pNP-0LB-tleap.mol2'

    def setup_method(self, method):
        super(TestGalbPNPMol2, self).setup_method(method)
        os.chdir('files')
        try: self.molecule = forcebalance.molecule.Molecule(self.source)
        except IOError:
            pytest.skip("Input gro file test/files/%s doesn't exist" % self.source)
        except:
            pytest.fail("Unable to open mol2 file")

    def test_read_galb(self):
        """Check for the correct number of bonds in a simple molecule"""
        self.logger.debug("\nTrying to read alanine dipeptide conformation... ")
        assert self.molecule.resname == 14 * ['PNP'] + 22 * ['0LB'], "Incorrect residue names"
        assert self.molecule.elem == ['O', 'N', 'O', 'C', 'C', 'C', 'H', 'H', 'C', 'H', 'C', 'H', 'C', 'O', 'C', 'H',
                                      'O', 'C', 'H', 'C', 'H', 'H', 'O', 'H', 'C', 'H', 'O', 'H', 'C', 'H', 'O', 'H',
                                      'C', 'H', 'O', 'H'], "Incorrect atomic symbols"
        assert len(self.molecule.bonds) == 37, "Incorrect number of bonds for pNP-0LB structure"
