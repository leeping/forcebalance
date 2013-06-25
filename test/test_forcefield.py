import sys, os
import forcebalance.forcefield as forcefield
import unittest
from __init__ import ForceBalanceTestCase, TestValues
import numpy as np

class TestWaterFF(ForceBalanceTestCase):
    """Test FF class using water options and forcefield"""
    def setUp(self):
        self.cwd=os.getcwd()
        os.chdir(TestValues.water_options['root'])
        self.ff = forcefield.FF(TestValues.water_options)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_FF_yields_consistent_results(self):
        """Check whether multiple calls to FF yield the same result"""
        self.assertTrue(forcefield.FF(TestValues.water_options)==forcefield.FF(TestValues.water_options))

    def test_make_function(self):
        """Check that make() function performs as expected"""
        pvals = self.ff.pvals0

        new_pvals = self.ff.make(np.zeros((3,3)))
        # given zero matrix, make should return unchanged pvals
        self.assertTrue((pvals == new_pvals).all())

        new_pvals = self.ff.make(np.random.rand(3,3))
        # given arbitrary nonzero input, make should return new pvals
        self.assertFalse((pvals == new_pvals).all(), msg="\nmake() returned unchanged pvals even when given nonzero matrix")

        random_array = np.random.rand(3,3).flatten()
        new_pvals = self.ff.make(random_array,use_pvals=True)
        self.assertTrue((random_array == new_pvals).all(), msg="\nmake() did not return input pvals with use_pvals=True")