"""Tests for AbInitioPairwise and AbInitioPairwise_SMIRNOFF targets."""
from __future__ import absolute_import

import os
import sys
import shutil

import numpy as np
import pytest

import forcebalance
import forcebalance.smirnoffio
from .__init__ import ForceBalanceTestCase
from .test_target import TargetTests
from .test_system import skip_openff_py39

has_openff_toolkit = True
try:
    import openff.toolkit
except ModuleNotFoundError:
    has_openff_toolkit = False

try:
    try:
        from openmm.app import *
        from openmm import *
        from openmm.unit import *
    except ImportError:
        from simtk.openmm.app import *
        from simtk.openmm import *
        from simtk.unit import *
    no_openmm = False
except ImportError:
    no_openmm = True


@skip_openff_py39
@pytest.mark.skipif(
    not has_openff_toolkit, reason="openff.toolkit not found"
)
@pytest.mark.skipif(no_openmm, reason="OpenMM not found")
class TestAbInitioPairwise_SMIRNOFF(TargetTests):
    """Test AbInitioPairwise_SMIRNOFF using a 5-conformation ethanol dataset."""

    def setup_method(self, method):
        super(TestAbInitioPairwise_SMIRNOFF, self).setup_method(method)
        self.options.update({
            'jobtype': 'NEWTON',
            'forcefield': ['ethanol-smirnoff.offxml'],
        })
        self.tgt_opt.update({
            'type': 'ABINITIOPAIRWISE_SMIRNOFF',
            'name': 'ethanol-pairwise',
            'mol2': ['ethanol.sdf'],
            'energy': True,
            'force': False,
            'w_energy': 1.0,
            'w_force': 0.0,
        })
        self.ff = forcebalance.forcefield.FF(self.options)
        self.mvals = np.array([0.0] * self.ff.np)

        self.target = forcebalance.smirnoffio.AbInitioPairwise_SMIRNOFF(
            self.options, self.tgt_opt, self.ff
        )

    def teardown_method(self):
        shutil.rmtree('temp', ignore_errors=True)
        super(TestAbInitioPairwise_SMIRNOFF, self).teardown_method()

    def test_force_raises(self):
        """Enabling force fitting should raise RuntimeError."""
        bad_opt = self.tgt_opt.copy()
        bad_opt['force'] = True
        with pytest.raises(RuntimeError):
            forcebalance.smirnoffio.AbInitioPairwise_SMIRNOFF(
                self.options, bad_opt, self.ff
            )

    def test_pairwise_pairs_count(self):
        """Number of pairs should be n*(n-1)/2 for n snapshots."""
        ns = self.target.ns
        expected_pairs = ns * (ns - 1) // 2
        assert len(self.target.eqm_pairs) == expected_pairs
        assert len(self.target.boltz_wt_pairs) == expected_pairs

    def test_pairwise_weights_sum_to_one(self):
        """Boltzmann weights for pairs should be normalized."""
        assert abs(self.target.boltz_wt_pairs.sum() - 1.0) < 1e-10

    def test_pairwise_energy_differences(self):
        """Pairwise QM energy differences should match manual computation."""
        import itertools
        eqm = self.target.eqm
        for i, (a, b) in enumerate(itertools.combinations(range(self.target.ns), 2)):
            expected = eqm[a] - eqm[b]
            assert abs(self.target.eqm_pairs[i] - expected) < 1e-10
