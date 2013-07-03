import sys, os
import forcebalance.forcefield as forcefield
import unittest
from __init__ import ForceBalanceTestCase, TestValues
import numpy as np
from copy import deepcopy

class TestWaterFF(ForceBalanceTestCase):
    """Test FF class using water options and forcefield (text forcefield input)"""
    def setUp(self):
        # options used in 001_water_tutorial
        self.options=TestValues.opts.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})
        os.chdir(self.options['root'])
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]

    def test_FF_yields_consistent_results(self):
        """Check whether multiple calls to FF yield the same result"""
        self.assertTrue(forcefield.FF(self.options)==forcefield.FF(self.options))

    def test_make_function_return_value(self):
        """Check that make() return value meets expectation"""
        pvals = self.ff.pvals0

        new_pvals = self.ff.make(np.zeros(self.ff.np))
        # given zero matrix, make should return unchanged pvals
        self.assertTrue((pvals == new_pvals).all())

        new_pvals = self.ff.make(np.ones(self.ff.np))
        # given arbitrary nonzero input, make should return new pvals
        self.assertFalse((pvals == new_pvals).all(), msg="\nmake() returned unchanged pvals even when given nonzero matrix")

        new_pvals = self.ff.make(np.ones(self.ff.np),use_pvals=True)
        self.assertTrue((np.ones(self.ff.np) == new_pvals).all(), msg="\nmake() did not return input pvals with use_pvals=True")

    def test_make_function_output(self):
        """Check make() function creates expected forcefield file"""

        self.ff.make(np.zeros(self.ff.np))
        os.rename(self.ff.fnms[0], self.options['ffdir']+'/test_zeros.' + self.filetype)
        self.options['forcefield']=['test_zeros.'+ self.filetype]
        ff_zeros = forcefield.FF(self.options)
        self.assertEqual(self.ff, ff_zeros,
                        msg = "make(0) produced a different output forcefield")
        os.remove(self.options['ffdir']+'/test_zeros.' + self.filetype)

        self.ff.make(np.ones(self.ff.np))
        os.rename(self.ff.fnms[0], self.options['ffdir']+'/test_ones.' + self.filetype)
        self.options['forcefield']=['test_ones.'+ self.filetype]
        ff_ones = forcefield.FF(self.options)
        self.assertNotEqual(self.ff, ff_ones,
                        msg = "make([1]) produced an unchanged output forcefield")
        os.remove(self.options['ffdir']+'/test_ones.' + self.filetype)

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return "ITP Forcefield: " + super(TestWaterFF,self).shortDescription()

class TestXmlFF(TestWaterFF):
    """Test FF class using water options and forcefield (text forcefield input)"""
    def setUp(self):
        # options from 2013 tutorial
        self.options=TestValues.opts.copy()
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
        return "XML Forcefield: " + super(TestWaterFF,self).shortDescription()

class TestGbsFF(TestWaterFF):
    """Test FF class using water options and forcefield (text forcefield input)"""
    def setUp(self):
        # options from 2013 tutorial
        self.options=TestValues.opts.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        os.chdir(self.options['root'])
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return "GBS Forcefield: " + super(TestWaterFF,self).shortDescription()
if __name__ == '__main__':           
    unittest.main()
