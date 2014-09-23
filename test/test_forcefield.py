import sys, os
import forcebalance
import forcebalance.forcefield as forcefield
import unittest
from __init__ import ForceBalanceTestCase
import numpy as np
from copy import deepcopy

class FFTests(object):
    """Tests common to all forcefields. Note that to prevent this class from being run on its own
    by the Test Runner, we do not subclass ForceBalanceTestCase. The actual forcefield instance
    being tested needs to be provided by subclasses"""

    def test_FF_yields_consistent_results(self):
        """Check whether multiple calls to FF yield the same result"""
        self.logger.debug("\nChecking consistency of ForceField constructor\n")
        self.assertEqual(forcefield.FF(self.options),forcefield.FF(self.options),
        msg = "\nGot two different forcefields despite using the same options as input")

    def test_make_function_return_value(self):
        """Check that make() return value meets expectation"""
        pvals = self.ff.pvals0

        self.logger.debug("Running forcefield.make() with zero vector should not change pvals... ")
        new_pvals = np.array(self.ff.make(np.zeros(self.ff.np)))
        self.assertEqual(pvals.size,new_pvals.size)
        # given zero matrix, make should return unchanged pvals
        self.assertEqual(pvals,new_pvals,
                    msg="\nmake() should produce unchanged pvals when given zero vector")
        self.logger.debug("ok\n")

        self.logger.debug("make() should return different values when passed in nonzero pval matrix... ")
        new_pvals = np.array(self.ff.make(np.ones(self.ff.np)))
        self.assertEqual(pvals.size,new_pvals.size)
        # given arbitrary nonzero input, make should return new pvals
        self.assertFalse((pvals==new_pvals).all(), msg="\nmake() returned unchanged pvals even when given nonzero matrix")
        self.logger.debug("ok\n")

        self.logger.debug("make(use_pvals=True) should return the same pvals... ")
        new_pvals = np.array(self.ff.make(np.ones(self.ff.np),use_pvals=True))
        self.assertEqual(np.ones(self.ff.np),new_pvals, msg="\nmake() did not return input pvals with use_pvals=True")
        self.logger.debug("ok\n")

        os.remove(self.options['root'] + '/' + self.ff.fnms[0])

    def test_make_function_output(self):
        """Check make() function creates expected forcefield file"""

        # read a forcefield from the output of make([--0--])
        self.ff.make(np.zeros(self.ff.np))
        os.rename(self.ff.fnms[0], self.options['ffdir']+'/test_zeros.' + self.filetype)
        self.options['forcefield']=['test_zeros.'+ self.filetype]
        ff_zeros = forcefield.FF(self.options)
        self.assertEqual(self.ff, ff_zeros,
                        msg = "make([0]) produced a different output forcefield")
        os.remove(self.options['ffdir']+'/test_zeros.' + self.filetype)

        # read a forcefield from the output of make([--1--])
        self.ff.make(np.ones(self.ff.np))
        os.rename(self.ff.fnms[0], self.options['ffdir']+'/test_ones.' + self.filetype)
        self.options['forcefield']=['test_ones.'+ self.filetype]
        ff_ones = forcefield.FF(self.options)

        self.assertNotEqual(self.ff, ff_ones,
                        msg = "make([1]) produced an unchanged output forcefield")
        os.remove(self.options['ffdir']+'/test_ones.' + self.filetype)
    

class TestWaterFF(ForceBalanceTestCase, FFTests):
    """Test FF class using water options and forcefield (text forcefield input)
    This test case also acts as a base class for other forcefield test cases.
    Override the setUp() to run tests on a different forcefield"""
    def setUp(self):
        # options used in 001_water_tutorial
        self.logger.debug("\nSetting up options...\n")
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})
        self.logger.debug(str(self.options) + '\n')
        os.chdir(self.options['root'])
        self.logger.debug("Creating forcefield using above options... ")
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.logger.debug("ok\n")

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestWaterFF,self).shortDescription() + " (itp)"

class TestXmlFF(ForceBalanceTestCase, FFTests):
    """Test FF class using dms.xml forcefield input"""
    def setUp(self):
        # options from 2013 tutorial
        self.logger.debug("Setting up options...\n")
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})
        self.logger.debug(str(self.options) + '\n')
        os.chdir(self.options['root'])

        self.logger.debug("Creating forcefield using above options... ")
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.logger.debug("ok\n")

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestXmlFF,self).shortDescription() + " (xml)"

class TestXmlScriptFF(ForceBalanceTestCase):
    """Test FF class with XmlScript using TIP3G2w.xml forcefield input"""
    def setUp(self):
        self.logger.debug("Setting up options...\n")
        os.chdir('test/files')
        # Load the base force field file
        self.ff = forcefield.FF.fromfile('forcefield/TIP3G2w.xml')
        # Load mathematical parameter values corresponding to a known output
        self.mvals = np.loadtxt('XmlScript_out/mvals.txt')
        # Load the known output force field
        self.ff_ref = forcefield.FF.fromfile('XmlScript_out/TIP3G2w_out_ref.xml')

    def tearDown(self):
        os.system('rm -rf TIP3G2w.xml')
        super(ForceBalanceTestCase,self).tearDown()

    def test_make_function_output(self):
        """Check make() function creates expected force field file containing XML Script"""
        os.chdir('XmlScript_out')
        # Create the force field with mathematical parameter values and
        # make sure it matches the known reference
        self.ff.make(self.mvals)
        ff_out = forcefield.FF.fromfile('TIP3G2w.xml')
        self.assertEqual(self.ff_ref, ff_out,
                        msg = "make() produced a different output force field")

class TestGbsFF(ForceBalanceTestCase, FFTests):
    """Test FF class using gbs forcefield input"""
    def setUp(self):
        self.logger.debug("Setting up options...\n")
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        self.logger.debug(str(self.options) + '\n')
        os.chdir(self.options['root'])

        self.logger.debug("Creating forcefield using above options... ")
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.logger.debug("ok\n")

    def test_find_spacings(self):
        """Check find_spacings function"""
        self.logger.debug("Running forcefield.find_spacings()...\n")
        spacings = self.ff.find_spacings()

        self.assertGreaterEqual((self.ff.np)*(self.ff.np-1)/2,len(spacings.keys()))
        self.assertEqual(dict, type(spacings))

    def shortDescription(self):
        """Add gbs to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestGbsFF,self).shortDescription() + " (gbs)"

if __name__ == '__main__':           
    unittest.main()
