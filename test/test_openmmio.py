import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase
from test_target import TargetTests # general targets tests defined in test_target.py
import logging

class TestLiquid_OpenMM(ForceBalanceTestCase, TargetTests):
    def setUp(self):
        self.skipTest("Needs optimizing to reduce runtime")
        TargetTests.setUp(self)
        # settings specific to this target
        self.options.update({
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})

        self.tgt_opt.update({'type':'LIQUID_OPENMM',
            'name':'dms-liquid'})

        self.ff = forcebalance.forcefield.FF(self.options)

        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.mvals = [.5]*self.ff.np

        self.target = forcebalance.openmmio.Liquid_OpenMM(self.options, self.tgt_opt, self.ff)
        self.target.stage(self.mvals)
        self.addCleanup(os.system, 'rm -rf temp')

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestLiquid_OpenMM,self).shortDescription() + " (Liquid_OpenMM)"

class TestInteraction_OpenMM(ForceBalanceTestCase, TargetTests):
    def setUp(self):
        TargetTests.setUp(self)
        # settings specific to this target
        self.options.update({
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})

        self.tgt_opt.update({"type" : "Interaction_OpenMM",
                            "name" : "S2EPose",
                            "fragment1" : "1-9",
                            "fragment2" : "10-18"})

        self.ff = forcebalance.forcefield.FF(self.options)

        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.mvals = [.5]*self.ff.np

        self.target = forcebalance.openmmio.Interaction_OpenMM(self.options, self.tgt_opt, self.ff)
        self.addCleanup(os.system, 'rm -rf temp')

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestInteraction_OpenMM,self).shortDescription() + " (Interaction_OpenMM)"

if __name__ == '__main__':           
    unittest.main()
