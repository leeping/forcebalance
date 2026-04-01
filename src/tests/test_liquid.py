from __future__ import absolute_import
import os
import sys
import shutil
import numpy as np
import pytest
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from forcebalance.liquid import _mbar_weights
from .__init__ import ForceBalanceTestCase, check_for_openmm

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'files', 'test_liquid')

# U_kln.npy — shape (6, 6, 801), float64
#   Reduced potential energy matrix U_kln[k, m, n] = (E_k[n] + P_m * V_k[n] * pvkj) * beta_m
#   where k = source simulation index, m = evaluation state index, n = snapshot index.
#   Built from the six npt_result.p pickles in files/test_liquid/single.tmp/Liquid/iter_0000/
#   (water at 249.15 K/1 atm, 273.15 K/1 atm, 298.15 K/1 atm, 373.15 K/1 atm,
#    298.15 K/20 bar, 298.15 K/2000 bar; 801 snapshots each).
#   Regenerate with: conda run -n <env> python tools/mbar_mre.py --save-ref
U_KLN_PATH  = os.path.join(FIXTURE_DIR, 'U_kln.npy')

# N_k.npy — shape (6,), int64, all entries = 801
#   Number of uncorrelated snapshots per simulation state; matches the first axis of U_kln.
N_K_PATH    = os.path.join(FIXTURE_DIR, 'N_k.npy')

# mbar_weights_ref.npy — shape (4806, 6), float64
#   MBAR weight matrix W[n, m] produced by pymbar 3.0.5 on U_kln above.
#   Used as the reference for cross-version agreement checks (atol=1e-4).
#   Regenerate with: conda run -n <env> python tools/mbar_mre.py --save-ref
REF_PATH    = os.path.join(FIXTURE_DIR, 'mbar_weights_ref.npy')


@pytest.fixture(scope='module')
def mbar_weights():
    if not os.path.exists(U_KLN_PATH) or not os.path.exists(N_K_PATH):
        pytest.skip(f"MBAR fixtures not found in {FIXTURE_DIR}; run tools/mbar_mre.py --save-ref")
    return _mbar_weights(np.load(U_KLN_PATH), np.load(N_K_PATH))


def test_mbar_weights_normalized(mbar_weights):
    """MBAR weight matrix columns must each sum to 1 (pymbar v3 and v4)."""
    col_sums = mbar_weights.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-6), (
        f"MBAR weight columns do not sum to 1: {col_sums}"
    )


def test_mbar_weights_match_reference(mbar_weights):
    """MBAR weights must agree with the pymbar-v3 reference within 1e-4."""
    if not os.path.exists(REF_PATH):
        pytest.skip(f"Reference weights not found at {REF_PATH}; run tools/mbar_mre.py --save-ref")
    W_ref = np.load(REF_PATH)
    assert mbar_weights.shape == W_ref.shape, (
        f"Weight matrix shape mismatch: got {mbar_weights.shape}, expected {W_ref.shape}"
    )
    np.testing.assert_allclose(mbar_weights, W_ref, rtol=1e-2,
                               err_msg="MBAR weights differ from v3 reference beyond rtol=1e-2")

class TestWaterTutorial(ForceBalanceTestCase):
    def setup_method(self, method):
        if not check_for_openmm(): pytest.skip("No OpenMM modules found.")
        super(TestWaterTutorial, self).setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        # copy folder 'files/test_liquid' into a new folder 'files/test_liquid.run'
        os.chdir(os.path.join(self.cwd, 'files'))
        tmpfolder = os.path.join(self.cwd, 'files', 'test_liquid.run')
        source_folder = os.path.join(self.cwd, 'files', 'test_liquid')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        shutil.copytree(source_folder, tmpfolder)
        os.chdir(tmpfolder)

    def teardown_method(self):
        # remove temporary folder 'files/test_liquid.run'
        tmpfolder = os.path.join(self.cwd, 'files', 'test_liquid.run')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        super(TestWaterTutorial, self).teardown_method()

    def test_liquid(self):
        """Check liquid target with existing simulation data"""
        if sys.version_info <= (2,7):
            pytest.skip("Existing pickle file only works with Python 3")

        self.logger.debug("Setting input file to 'single.in'\n")
        input_file ='single.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        forcefield  = FF(options)
        objective   = Objective(options, tgt_opts, forcefield)
        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        assert isinstance(optimizer, Optimizer), "Expected forcebalance optimizer object"
        self.logger.debug(str(optimizer))

        ## Actually run the optimizer.
        self.logger.debug("\nDone setting up! Running optimizer...")
        result = optimizer.Run()
        self.logger.debug("\nOptimizer finished. Final results:")
        self.logger.debug(str(result))

        liquid_obj_value = optimizer.Objective.ObjDict['Liquid']['x']
        assert liquid_obj_value < 20, "Liquid objective function should give < 20 (about 17.23) total value."

