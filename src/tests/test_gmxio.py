from __future__ import absolute_import
import os
import forcebalance
import shutil
import pytest
from forcebalance.nifty import *
from forcebalance.gmxio import GMX
from .test_target import TargetTests # general targets tests defined in test_target.py
"""
The testing functions for this class are located in test_target.py.
"""

class TestAbInitio_GMX(TargetTests):
    def setup_method(self, method):
        super(TestAbInitio_GMX, self).setup_method(method)
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

    def teardown_method(self):
        # Use an absolute path so this works even if the test left us inside
        # the temp directory (e.g. because it errored before os.chdir('../..')).
        temp_dir = os.path.join(os.path.dirname(__file__), 'files', 'temp')
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
        super(TestAbInitio_GMX, self).teardown_method()


@pytest.mark.gmx_rejected
def test_gmx_version_rejected():
    """
    GMX() must raise RuntimeError when a known-bad GROMACS version is installed.
    Skipped by default; enable with `pytest --run-gmx-rejected`.
    """
    with pytest.raises(RuntimeError):
        GMX()


def test_gmx_version_accepted(monkeypatch):
    """GMX() must not raise a version error for supported GROMACS versions (5.x, 2024.4+)."""
    monkeypatch.setattr(GMX, 'readsrc', lambda self, **kw: None)
    monkeypatch.setattr(GMX, 'prepare', lambda self, **kw: None)
    GMX()  # version check passes; file I/O is monkeypatched out

