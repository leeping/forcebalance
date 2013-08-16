import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase
from test_target import TargetTests # general targets tests defined in test_target.py

class TestInteraction_TINKER(ForceBalanceTestCase, TargetTests):
    def setUp(self):
        TargetTests.setUp(self)
        self.options.update({
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cl4.prm']})

        self.tgt_opt.update({'type':'Interaction_TINKER',
                             'name':'ccl4-h2o-1',
                             'energy_denom':1.0,
                             'attenuate':'True',
                             'fragment1':'1-5',
                             'fragment2':'6-8',
                             'energy-upper':20.0})
                             
        self.logger.debug("\nOptions:\n%s\n" % str(self.options))
        self.logger.debug("\nTarget Options:\n%s\n" % str(self.tgt_opt))
                             
        if not os.path.exists(self.options['tinkerpath']): self.skipTest("options['tinkerpath'] is not a valid path")

        try:
            self.ff = forcebalance.forcefield.FF(self.options)
        except:
            self.skipTest("Unable to create forcefield from cl4.prm\n")

        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.mvals = [.5]*self.ff.np

        self.logger.debug("Setting up Interaction_TINKER target\n")
        self.target = forcebalance.tinkerio.Interaction_TINKER(self.options, self.tgt_opt, self.ff)
        self.addCleanup(os.system, 'rm -rf temp')

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestInteraction_TINKER,self).shortDescription() + " (Interaction_TINKER)"

if __name__ == '__main__':           
    unittest.main()
