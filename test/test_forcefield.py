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
        self.assertEqual(forcefield.FF(self.options),forcefield.FF(self.options),
        msg = "\nGot two different forcefields despite using the same options")

    def test_make_function_return_value(self):
        """Check that make() return value meets expectation"""
        pvals = self.ff.pvals0

        new_pvals = np.array(self.ff.make(np.zeros(self.ff.np)))
        self.assertEqual(pvals.size,new_pvals.size)
        # given zero matrix, make should return unchanged pvals
        self.assertEqual(pvals,new_pvals,
        msg="\nmake() should produce unchanged pvals when given zero vector")

        new_pvals = np.array(self.ff.make(np.ones(self.ff.np)))
        self.assertEqual(pvals.size,new_pvals.size)
        # given arbitrary nonzero input, make should return new pvals
        self.assertFalse((pvals==new_pvals).all(), msg="\nmake() returned unchanged pvals even when given nonzero matrix")

        new_pvals = np.array(self.ff.make(np.ones(self.ff.np),use_pvals=True))
        self.assertEqual(np.ones(self.ff.np),new_pvals, msg="\nmake() did not return input pvals with use_pvals=True")

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
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})
        os.chdir(self.options['root'])
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestWaterFF,self).shortDescription() + " (itp)"

class TestXmlFF(ForceBalanceTestCase, FFTests):
    """Test FF class using dms.xml forcefield input"""
    def setUp(self):
        # options from 2013 tutorial
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})
        os.chdir(self.options['root'])
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestXmlFF,self).shortDescription() + " (xml)"

class TestGbsFF(ForceBalanceTestCase, FFTests):
    """Test FF class using gbs forcefield input"""
    def setUp(self):
        # options from 2013 tutorial
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        os.chdir(self.options['root'])
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]

    def test_find_spacings(self):
        """Check find_spacings function"""
        spacings = self.ff.find_spacings()

        self.assertGreaterEqual((self.ff.np)*(self.ff.np-1)/2,len(spacings.keys()))
        self.assertEqual(dict, type(spacings))

    def shortDescription(self):
        """Add gbs to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestGbsFF,self).shortDescription() + " (gbs)"

if __name__ == '__main__':           
    unittest.main()
