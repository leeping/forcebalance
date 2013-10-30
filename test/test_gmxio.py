import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase
from test_target import TargetTests # general targets tests defined in test_target.py

class TestAbInitio_GMX(ForceBalanceTestCase, TargetTests):
    def setUp(self):
        TargetTests.setUp(self)
        self.options.update({
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})

        self.tgt_opt.update({'type':'ABINITIO_GMX',
            'name':'cluster-02'})

        self.ff = forcebalance.forcefield.FF(self.options)

        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.mvals = [.5]*self.ff.np

        self.logger.debug("Setting up AbInitio_GMX target\n")
        self.target = forcebalance.gmxio.AbInitio_GMX(self.options, self.tgt_opt, self.ff)
        self.addCleanup(os.system, 'rm -rf temp')

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestAbInitio_GMX,self).shortDescription() + " (AbInitio_GMX)"

if __name__ == '__main__':           
    unittest.main()
