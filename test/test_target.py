import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase

class TargetTests(object):
    def setUp(self):
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.tgt_opt=forcebalance.parser.tgt_opts_defaults.copy()
        self.ff = None  # Forcefield this target is fitting
        self.options.update({'root': os.getcwd() + '/test/files'})

        os.chdir(self.options['root'])


    def test_get_function(self):
        """Check target get() function output"""
        os.chdir('temp/%s' % self.tgt_opt['name'])
        objective = self.target.get([1]*self.ff.np)
        
        # check objective dictionary keys
        self.assertEqual(dict,type(objective))
        self.assertTrue(objective.has_key('X'))
        self.assertTrue(objective.has_key('G'))
        self.assertTrue(objective.has_key('H'))

        # check objective value types
        self.assertEqual(numpy.float64, type(objective['X']))
        self.assertEqual(numpy.ndarray, type(objective['G']))
        self.assertEqual(numpy.ndarray, type(objective['H']))

        # check array dimensions
        self.assertEqual(objective['G'].size, self.ff.np)
        self.assertEqual(objective['H'].size, self.ff.np**2)
        self.assertEqual(objective['G'].ndim, 1)
        self.assertEqual(objective['H'].ndim, 2)
        self.assertEqual(objective['G'].shape, (self.ff.np,))
        self.assertEqual(objective['H'].shape, (self.ff.np, self.ff.np))

        os.chdir('../..')

    def test_get_agrad(self):
        """Check target objective function gradient using finite difference"""
        self.mvals = [.5]*self.ff.np

        os.chdir('temp/%s' % self.tgt_opt['name'])

        objective = self.target.get(self.mvals, AGrad=True)
        X=objective['X']
        G=objective['G']
        self.assertTrue(G.any())    # with AGrad=True, G should not be [0]
        g=numpy.zeros(self.ff.np)

        for p in range(self.ff.np):
            mvals_lo = self.mvals[:]
            mvals_hi = self.mvals[:]
            mvals_lo[p]-=(self.mvals[p]/200.)
            mvals_hi[p]+=(self.mvals[p]/200.)

            Xlo = self.target.get(mvals_lo)['X']
            Xhi = self.target.get(mvals_hi)['X']
            g[p] = (Xhi-Xlo)/(self.mvals[p]/100.)
            #sys.stderr.write("\n%f\t%f" % (G[p],g[p]))
            self.assertAlmostEqual(g[p], G[p], delta=X*.01)

        os.chdir('../..')
        

if __name__ == '__main__':           
    unittest.main()
