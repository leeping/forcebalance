from __future__ import absolute_import
# import unittest
import sys, os, re
import forcebalance
import shutil
import abc
import numpy
# from .__init__ import ForceBalanceTestCase
from .test_target import TargetTests # general targets tests defined in test_target.py
import logging
import pytest
"""
The testing functions for this class are located in test_target.py.
"""
# class TestLiquid_OpenMM(ForceBalanceTestCase, TargetTests):
class TestLiquid_OpenMM(TargetTests):
    # def setUp(self):
    def setup_method(self, method):
        # super(TestLiquid_OpenMM, cls).setup_class()
        pytest.skip("Needs optimizing to reduce runtime")
        super(TestLiquid_OpenMM, self).setup_method(method)# .im_func()
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
        #pytest.addCleanup(os.system, 'rm -rf temp')

    def teardown_method(self):
        shutil.rmtree('temp')
        super(TestLiquid_OpenMM, self).teardown_method()

    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestLiquid_OpenMM,self).shortDescription() + " (Liquid_OpenMM)"

# class TestInteraction_OpenMM(ForceBalanceTestCase, TargetTests):
class TestInteraction_OpenMM(TargetTests):
    # def setUp(self):

    def setup_method(self, method):
        super(TestInteraction_OpenMM, self).setup_method(method)
        # TargetTests.setup_class(cls)
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

    def teardown_method(self):
        #os.system('rm -rf temp')
        shutil.rmtree('temp')
        super(TestInteraction_OpenMM, self).teardown_method()


    def shortDescription(self):
        """@override ForceBalanceTestCase.shortDescription()"""
        return super(TestInteraction_OpenMM,self).shortDescription() + " (Interaction_OpenMM)"
