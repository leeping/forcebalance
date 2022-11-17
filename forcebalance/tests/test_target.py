from __future__ import division
from __future__ import absolute_import
from builtins import str
from builtins import range
import os
import forcebalance
import numpy
from .__init__ import ForceBalanceTestCase

class TargetTests(ForceBalanceTestCase):

    def setup_method(self, method):
        super(TargetTests, self).setup_method(method)
        self.logger = forcebalance.output.getLogger('forcebalance.test.' + __name__[5:])
        self.logger.debug("\nBuilding options for target...\n")
        self.options = forcebalance.parser.gen_opts_defaults.copy()
        self.tgt_opt = forcebalance.parser.tgt_opts_defaults.copy()
        self.ff = None  # Forcefield this target is fitting
        self.options.update({'root': os.path.join(os.getcwd(), 'files')})
        self.check_grad_fd = True # Whether to check gradient vs. finite difference. Set to False for liquid targets.

        os.chdir(self.options['root'])

    def test_get_function(self):
        """Check target get() function output"""
        #os.chdir(self.target.tempdir)
        os.chdir('temp/%s' % self.tgt_opt['name'])

        self.logger.debug("Evaluating objective function for target...\n")
        objective = self.target.get(self.mvals)
        self.target.indicate()
        print("objective =\n%s" % str(objective))

        # check objective dictionary keys
        print("\n>ASSERT objective dictionary has X, G, H keys\n")
        assert isinstance(objective, dict)
        assert 'X' in objective
        assert 'G' in objective
        assert 'H' in objective

        # check objective value types
        print(">ASSERT objective['X'] is a float\n")
        assert isinstance(objective['X'], numpy.float64)
        print(">ASSERT objective['G'] is a numpy array\n")
        assert isinstance(objective['G'], numpy.ndarray)
        print(">ASSERT objective['H'] is a numpy array\n")
        assert isinstance(objective['H'], numpy.ndarray)

        # check array dimensions
        print(">ASSERT size of objective['G'] is a equal to number of forcefield parameters (p)\n")
        assert objective['G'].size == self.ff.np
        print(">ASSERT size of objective['H'] is a equal to number of forcefield parameters squared (p^2)\n")
        assert objective['H'].size == self.ff.np**2
        print(">ASSERT objective['G'] is one dimensional\n")
        assert objective['G'].ndim == 1
        print(">ASSERT objective['H'] is two dimensional\n")
        assert objective['H'].ndim == 2
        print(">ASSERT objective['G'] is p x 1 array\n")
        assert objective['G'].shape == (self.ff.np,)
        print(">ASSERT objective['G'] is p x p array\n")
        assert objective['H'].shape == (self.ff.np, self.ff.np)

        os.chdir('../..')

    def test_get_agrad(self):
        """Check target objective function gradient using finite difference"""
        self.mvals = [.5]*self.ff.np

        os.chdir(os.path.join('temp', self.tgt_opt['name']))

        print("Running target.get(mvals, AGrad=True)\n")
        objective = self.target.get(self.mvals, AGrad=True)
        X = objective['X']
        G = objective['G']
        print(">ASSERT objective['G'] is not a zero vector\n")
        assert G.any() # with AGrad=True, G should not be [0]
        g = numpy.zeros(self.ff.np)

        if self.check_grad_fd:
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
    
        os.chdir('../..')
