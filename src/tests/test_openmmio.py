from __future__ import absolute_import
import forcebalance
import shutil
from .test_target import TargetTests # general targets tests defined in test_target.py
import pytest
"""
The testing functions for this class are located in test_target.py.
"""
class TestLiquid_OpenMM(TargetTests):
    def setup_method(self, method):
        pytest.skip("Needs optimizing to reduce runtime")
        super(TestLiquid_OpenMM, self).setup_method(method)
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

    def teardown_method(self):
        shutil.rmtree('temp')
        super(TestLiquid_OpenMM, self).teardown_method()

class TestInteraction_OpenMM(TargetTests):

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
        shutil.rmtree('temp')
        super(TestInteraction_OpenMM, self).teardown_method()

