import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase

class TargetTests(object):
    def setUp(self):
        self.logger.debug("\nBuilding options for target...\n")
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.tgt_opt=forcebalance.parser.tgt_opts_defaults.copy()
        self.ff = None  # Forcefield this target is fitting
        self.options.update({'root': os.getcwd() + '/test/files'})

        os.chdir(self.options['root'])

    def test_get_function(self):
        """Check target get() function output"""
        # os.chdir('temp/%s' % self.tgt_opt['name'])
        os.chdir(self.target.tempdir)

        self.logger.debug("Evaluating objective function for target...\n")
        objective = self.target.get(self.mvals)
        self.target.indicate()
        self.logger.debug("objective =\n%s" % str(objective))
        
        # check objective dictionary keys
        self.logger.debug("\n>ASSERT objective dictionary has X, G, H keys\n")
        self.assertEqual(dict,type(objective))
        self.assertTrue(objective.has_key('X'))
        self.assertTrue(objective.has_key('G'))
        self.assertTrue(objective.has_key('H'))

        # check objective value types
        self.logger.debug(">ASSERT objective['X'] is a float\n")
        self.assertEqual(numpy.float64, type(objective['X']))
        self.logger.debug(">ASSERT objective['G'] is a numpy array\n")
        self.assertEqual(numpy.ndarray, type(objective['G']))
        self.logger.debug(">ASSERT objective['H'] is a numpy array\n")
        self.assertEqual(numpy.ndarray, type(objective['H']))

        # check array dimensions
        self.logger.debug(">ASSERT size of objective['G'] is a equal to number of forcefield parameters (p)\n")
        self.assertEqual(objective['G'].size, self.ff.np)
        self.logger.debug(">ASSERT size of objective['H'] is a equal to number of forcefield parameters squared (p^2)\n")
        self.assertEqual(objective['H'].size, self.ff.np**2)
        self.logger.debug(">ASSERT objective['G'] is one dimensional\n")
        self.assertEqual(objective['G'].ndim, 1)
        self.logger.debug(">ASSERT objective['H'] is two dimensional\n")
        self.assertEqual(objective['H'].ndim, 2)
        self.logger.debug(">ASSERT objective['G'] is p x 1 array\n")
        self.assertEqual(objective['G'].shape, (self.ff.np,))
        self.logger.debug(">ASSERT objective['G'] is p x p array\n")
        self.assertEqual(objective['H'].shape, (self.ff.np, self.ff.np))

        os.chdir('../..')

    def test_get_agrad(self):
        """Check target objective function gradient using finite difference"""
        self.mvals = [.5]*self.ff.np

        os.chdir('temp/%s' % self.tgt_opt['name'])

        self.logger.debug("Running target.get(mvals, AGrad=True)\n")
        objective = self.target.get(self.mvals, AGrad=True)
        X=objective['X']
        G=objective['G']
        self.logger.debug(">ASSERT objective['G'] is not a zero vector\n")
        self.assertTrue(G.any())    # with AGrad=True, G should not be [0]
        g=numpy.zeros(self.ff.np)

        self.logger.debug(">ASSERT objective['G'] approximately matches finite difference calculations\n")
        for p in range(self.ff.np):
            mvals_lo = self.mvals[:]
            mvals_hi = self.mvals[:]
            mvals_lo[p]-=(self.mvals[p]/200.)
            mvals_hi[p]+=(self.mvals[p]/200.)

            Xlo = self.target.get(mvals_lo)['X']
            Xhi = self.target.get(mvals_hi)['X']
            g[p] = (Xhi-Xlo)/(self.mvals[p]/100.)
            self.assertAlmostEqual(g[p], G[p], delta=X*.01)

        os.chdir('../..')
        

if __name__ == '__main__':           
    unittest.main()
