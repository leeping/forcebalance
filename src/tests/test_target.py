from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
# import unittest
import sys, os, re
import forcebalance
import abc
import numpy
# from .__init__ import ForceBalanceTestCase

# class TargetTests(object):
class TargetTests(object):
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        print("\nBuilding options for target...\n")
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.tgt_opt=forcebalance.parser.tgt_opts_defaults.copy()
        cls.ff = None  # Forcefield this target is fitting
        cls.options.update({'root': os.getcwd()})

    def test_get_function(self):
        """Check target get() function output"""
        # os.chdir('temp/%s' % self.tgt_opt['name'])
        os.chdir(self.target.tempdir)

        print("Evaluating objective function for target...\n")
        objective = self.target.get(self.mvals)
        self.target.indicate()
        print("objective =\n%s" % str(objective))

        # check objective dictionary keys
        print("\n>ASSERT objective dictionary has X, G, H keys\n")
        assert isinstance(objective, dict)
        assert 'X' in objective
        assert 'G' in objective
        assert 'H' in objective
        # self.assertEqual(dict,type(objective))
        # self.assertTrue('X' in objective)
        # self.assertTrue('G' in objective)
        # self.assertTrue('H' in objective)

        # check objective value types
        print(">ASSERT objective['X'] is a float\n")
        assert isinstance(objective['X'], float64)
        # self.assertEqual(numpy.float64, type(objective['X']))
        print(">ASSERT objective['G'] is a numpy array\n")
        assert isinstance(objective['G'], float64)
        # self.assertEqual(numpy.ndarray, type(objective['G']))
        print(">ASSERT objective['H'] is a numpy array\n")
        assert isinstance(objective['H'], float64)
        # self.assertEqual(numpy.ndarray, type(objective['H']))

        # check array dimensions
        print(">ASSERT size of objective['G'] is a equal to number of forcefield parameters (p)\n")
        assert objective['G'].size == self.ff.n
        # self.assertEqual(objective['G'].size, self.ff.np)
        print(">ASSERT size of objective['H'] is a equal to number of forcefield parameters squared (p^2)\n")
        assert objective['H'].size == self.ff.np**2
        # self.assertEqual(objective['H'].size, self.ff.np**2)
        print(">ASSERT objective['G'] is one dimensional\n")
        assert objective['G'].ndim == 1
        # self.assertEqual(objective['G'].ndim, 1)
        print(">ASSERT objective['H'] is two dimensional\n")
        assert objective['H'].ndim == 2
        # self.assertEqual(objective['H'].ndim, 2)
        print(">ASSERT objective['G'] is p x 1 array\n")
        assert objective['G'].shape == (self.ff.np,)
        # self.assertEqual(objective['G'].shape, (self.ff.np,))
        print(">ASSERT objective['G'] is p x p array\n")
        assert objective['H'].shape == (self.ff.np, self.ff.np)
        # self.assertEqual(objective['H'].shape, (self.ff.np, self.ff.np))

        os.chdir('../..')

    def test_get_agrad(self):
        """Check target objective function gradient using finite difference"""
        self.mvals = [.5]*self.ff.np

        os.chdir('temp/%s' % self.tgt_opt['name'])

        print("Running target.get(mvals, AGrad=True)\n")
        objective = self.target.get(self.mvals, AGrad=True)
        X=objective['X']
        G=objective['G']
        print(">ASSERT objective['G'] is not a zero vector\n")
        assert G.any()
        # self.assertTrue(G.any())    # with AGrad=True, G should not be [0]
        g=numpy.zeros(self.ff.np)

        print(">ASSERT objective['G'] approximately matches finite difference calculations\n")
        for p in range(self.ff.np):
            mvals_lo = self.mvals[:]
            mvals_hi = self.mvals[:]
            mvals_lo[p]-=(self.mvals[p]/200.)
            mvals_hi[p]+=(self.mvals[p]/200.)

            Xlo = self.target.get(mvals_lo)['X']
            Xhi = self.target.get(mvals_hi)['X']
            g[p] = (Xhi-Xlo)/(self.mvals[p]/100.)
            assert abs(g[p]-G[p]) < X*.01 +1e-7
            # self.assertAlmostEqual(g[p], G[p], delta=X*.01)

        os.chdir('../..')


# if __name__ == '__main__':
#     unittest.main()
