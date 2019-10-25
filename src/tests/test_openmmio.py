from __future__ import absolute_import
# import unittest
import sys, os, re
import forcebalance
import abc
import numpy
# from .__init__ import ForceBalanceTestCase
from .test_target import TargetTests # general targets tests defined in test_target.py
import logging
import pytest
# class TestLiquid_OpenMM(ForceBalanceTestCase, TargetTests):
class TestLiquid_OpenMM(TargetTests):
    # def setUp(self):
    @classmethod
    def setup_class(cls):
        # super(TestLiquid_OpenMM, cls).setup_class()
        # self.skipTest("Needs optimizing to reduce runtime")
        TargetTests.setup_class.im_func()
        # settings specific to this target
        cls.options.update({
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})

        cls.tgt_opt.update({'type':'LIQUID_OPENMM',
            'name':'dms-liquid'})

        cls.ff = forcebalance.forcefield.FF(cls.options)

        cls.ffname = cls.options['forcefield'][0][:-3]
        cls.filetype = cls.options['forcefield'][0][-3:]
        cls.mvals = [.5]*cls.ff.np

        cls.target = forcebalance.openmmio.Liquid_OpenMM(cls.options, cls.tgt_opt, cls.ff)
        cls.target.stage(cls.mvals)
        pytest.addCleanup(os.system, 'rm -rf temp')

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestLiquid_OpenMM,self).shortDescription() + " (Liquid_OpenMM)"

# class TestInteraction_OpenMM(ForceBalanceTestCase, TargetTests):
class TestInteraction_OpenMM(TargetTests):
    # def setUp(self):
    @classmethod
    def setup_class(cls):
        TargetTests.setup_class.im_func()
        # TargetTests.setup_class(cls)
        # settings specific to this target
        cls.options.update({
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})

        cls.tgt_opt.update({"type" : "Interaction_OpenMM",
                            "name" : "S2EPose",
                            "fragment1" : "1-9",
                            "fragment2" : "10-18"})

        cls.ff = forcebalance.forcefield.FF(cls.options)

        cls.ffname = cls.options['forcefield'][0][:-3]
        cls.filetype = cls.options['forcefield'][0][-3:]
        cls.mvals = [.5]*cls.ff.np

        cls.target = forcebalance.openmmio.Interaction_OpenMM(cls.options, cls.tgt_opt, cls.ff)
        pytest.addCleanup(os.system, 'rm -rf temp')

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestInteraction_OpenMM,self).shortDescription() + " (Interaction_OpenMM)"

# if __name__ == '__main__':
#     unittest.main()
