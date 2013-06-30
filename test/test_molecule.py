import unittest
import sys, os, re
import forcebalance.molecule
from __init__ import ForceBalanceTestCase
import numpy as np

class TestPDBMolecule(ForceBalanceTestCase):
    def __init__(self, methodName='runTest'):
        super(TestPDBMolecule,self).__init__(methodName)
        self.source = 'dms_conf.pdb'

    def setUp(self):
        super(TestPDBMolecule,self).setUp()
        os.chdir('test/files')
        try: self.molecule = forcebalance.molecule.Molecule(self.source, build_topology=False)
        except IOError:
            self.skipTest("Input pdb file test/files/%s doesn't exist" % self.source)
        except:
            self.fail("\nUnable to open pdb file")

    def tearDown(self):
        os.system('rm -rf *.xyz *.gro')
        super(TestPDBMolecule,self).tearDown()

    def test_xyz_conversion(self):
        """Check molecule conversion from pdb to xyz format"""
        self.molecule.write(self.source[:-3] + '.xyz')
        try:
            molecule1 = forcebalance.molecule.Molecule(self.source[:-3] + '.xyz', build_topology=False)
        except:
            self.fail("\nConversion to xyz format creates unreadable file")

        self.assertTrue((self.molecule.Data['xyzs'][0]==molecule1.Data['xyzs'][0]).all(),
                        msg = "\nConversion from pdb to xyz yields different xyz coordinates")

    def test_gro_conversion(self):
        """Check molecule conversion from pdb to gro format"""
        self.molecule.write(self.source[:-3] + '.gro')
        try:
            molecule1 = forcebalance.molecule.Molecule(self.source[:-3] + '.gro', build_topology=False)
        except:
            self.fail("\nConversion to gro format creates unreadable file")

        self.assertEqual(len(self.molecule.Data['resid']), len(molecule1.Data['resid']),
                        msg = "\nConversion from pdb to gro yields different number of residues")

        self.assertTrue((abs(self.molecule.Data['xyzs'][0]-molecule1.Data['xyzs'][0])<.0000001).all(),
        msg = "\nConversion from pdb to gro yields different xyz coordinates\npdb:\n%s\n\ngro:\n%s\n" %\
        (str(self.molecule.Data['xyzs'][0]), str(molecule1.Data['xyzs'][0])))






if __name__ == '__main__':
    unittest.main()
