from __future__ import absolute_import

from builtins import str
import os, shutil
import sys
import tarfile
from .__init__ import ForceBalanceTestCase, check_for_openmm
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer, Counter
from numpy import array
import numpy as np
import pytest

skip_openff_py39 = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="openff packages require ambertools which requires Python >= 3.10",
)

# expected results (mvals) taken from previous runs. Update this if it changes and seems reasonable (updated 10/24/13)
#EXPECTED_WATER_RESULTS = array([3.3192e-02, 4.3287e-02, 5.5072e-03, -4.5933e-02, 1.5499e-02, -3.7655e-01, 2.4720e-03, 1.1914e-02, 1.5066e-01])
EXPECTED_WATER_RESULTS = array([4.2370e-02, 3.1217e-02, 5.6925e-03, -4.8114e-02, 1.6735e-02, -4.1722e-01, 6.2716e-03, 4.6306e-03, 2.5960e-01])

# expected results (mvals) taken from previous runs. Update this if it changes and seems reasonable (updated 01/24/14)
EXPECTED_BROMINE_RESULTS = array([-0.305718, -0.12497])

# expected objective function from 003d evaluator bromine study. (updated 11/23/19)
EXPECTED_EVALUATOR_BROMINE_OBJECTIVE = array([1000])

# expected gradient elements from 003d evaluator bromine study. Very large uncertainties of +/- 2000 (updated 11/23/19)
EXPECTED_EVALUATOR_BROMINE_GRADIENT = array([4500, 5500])

# expected objective values from 029 cross-engine consistency study. Update after first run.
EXPECTED_TIP4P_OBJECTIVE = array([0.473572])
EXPECTED_TIP5P_OBJECTIVE = array([0.360085])

# expected result (pvals) taken from ethanol GB parameter optimization. Update this if it changes and seems reasonable (updated 09/05/14)
EXPECTED_ETHANOL_RESULTS = array([1.2286e-01, 8.3624e-01, 1.0014e-01, 8.4533e-01, 1.8740e-01, 6.8820e-01, 1.4606e-01, 8.3518e-01])

# fail test if we take more than this many iterations to converge. Update this as necessary
ITERATIONS_TO_CONVERGE = 5

# expected results taken from previous runs. Update this if it changes and seems reasonable (updated 07/23/14)
EXPECTED_LIPID_RESULTS = array([-6.7553e-03, -2.4070e-02])

# expected results taken from OpenFF torsion profile optimization using OpenFF toolkit 0.4.1 and OpenEye toolkit 2019.10.2. (updated 02/06/23)
# EXPECTED_OPENFF_TORSIONPROFILE_RESULTS = array([-9.4238e-02, 7.3350e-03, -7.9467e-05, 1.7172e-02, -1.3309e-01, 6.0076e-02, 1.7895e-02, 6.5866e-02, -1.4084e-01, -2.2906e-02])
# 02/06/23: As of toolkit v0.11, the default charge assignment method changed, which caused the following change in the optimization result:
EXPECTED_OPENFF_TORSIONPROFILE_RESULTS = array([-8.6810e-02, 6.7106e-03, 3.0992e-03, 1.8605e-02, -1.1292e-01, 5.6741e-02, 1.8884e-02, 7.3325e-02, -1.4203e-01, -9.2920e-03])

# expected objective function from 025 recharge methane study. (updated 08/04/20)
EXPECTED_RECHARGE_METHANE_ESP_OBJECTIVE = array([5.68107e-04])
EXPECTED_RECHARGE_METHANE_FIELD_OBJECTIVE = array([7.43711e-04])

# expected gradient elements from 025 recharge methane. (updated 08/04/20)
EXPECTED_RECHARGE_METHANE_ESP_GRADIENT = array([9.76931016e-03])
EXPECTED_RECHARGE_METHANE_FIELD_GRADIENT = array([1.12071584e-02])

# in practice these aren't hit, we don't simulate nearly long enough
EXPECTED_VSITE_VDW_PARAMETERS = array([
    # CX4 epsilon, sigma
    0.1088406109251, 3.3795317616266205,
    # water O epsilon, sigma
    0.7492790213533, 0.3165552430462,
    # vsite distance, charge increment2
    -0.010527445756662016, 0.5258681106763
])


class ForceBalanceSystemTest(ForceBalanceTestCase):
    def teardown_method(self):
        for fnm in [self.input_file.replace('.in','.sav')]:
            if os.path.exists(fnm):
                os.remove(fnm)
        for dnm in [self.input_file.replace('.in','.bak'), self.input_file.replace('.in','.tmp'), "result"]:
            if os.path.exists(dnm):
                shutil.rmtree(dnm)
        super(ForceBalanceSystemTest, self).teardown_method()
        
    def get_objective(self):
        """ Return the objective function object """
        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(self.input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))
        assert isinstance(options, dict), "Parser gave incorrect type for options"
        assert isinstance(tgt_opts, list), "Parser gave incorrect type for tgt_opts"
        for target in tgt_opts:
            assert isinstance(target, dict), "Parser gave incorrect type for target dict"
        ## The force field component of the project
        forcefield  = FF(options)
        assert isinstance(forcefield, FF), "Expected forcebalance forcefield object"
        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        assert isinstance(objective, Objective), "Expected forcebalance objective object"
        return objective

    def get_optimizer(self):
        """ Return the optimizer object """
        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(self.input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))
        assert isinstance(options, dict), "Parser gave incorrect type for options"
        assert isinstance(tgt_opts, list), "Parser gave incorrect type for tgt_opts"
        for target in tgt_opts:
            assert isinstance(target, dict), "Parser gave incorrect type for target dict"
        ## The force field component of the project
        forcefield  = FF(options)
        assert isinstance(forcefield, FF), "Expected forcebalance forcefield object"
        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        assert isinstance(objective, Objective), "Expected forcebalance objective object"
        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        assert isinstance(optimizer, Optimizer), "Expected forcebalance optimizer object"
        self.logger.debug(str(optimizer) + "\n")
        return optimizer

    def run_optimizer(self, check_result=True, check_iter=True, use_pvals=False):
        optimizer = self.get_optimizer()
        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()
        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')
        ## Convert result to physical values if desired.
        if use_pvals:
            result = optimizer.FF.create_pvals(result)
        if check_result:
            msg = "\nCalculation results have changed from previously calculated values.\n " \
                  "If this seems reasonable, update %s in test_system.py with these values" % self.expected_results_name
            np.testing.assert_allclose(self.expected_results, result, atol=self.absolute_tolerance, err_msg=msg)
        if check_iter:
            # Fail if calculation takes longer than previously to converge
            assert ITERATIONS_TO_CONVERGE >= Counter(), "Calculation took longer than expected to converge (%d iterations vs previous of %d)" %\
                (ITERATIONS_TO_CONVERGE, Counter())
        return result

class TestWaterTutorial(ForceBalanceSystemTest):
    def setup_method(self, method):
        super(TestWaterTutorial, self).setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, '..','..', 'studies','001_water_tutorial'))
        targets = tarfile.open('targets.tar.bz2','r')
        targets.extractall()
        targets.close()
        self.input_file='very_simple.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_WATER_RESULTS"
        self.expected_results = EXPECTED_WATER_RESULTS
        self.absolute_tolerance = 0.005

    def test_water_tutorial(self):
        """Check water tutorial study runs without errors"""
        self.run_optimizer()


class TestVoelzStudy(ForceBalanceSystemTest):
    def setup_method(self, method):
        super(TestVoelzStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '009_voelz_nspe'))
        self.input_file='options.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)

    def test_voelz_study(self):
        """Check voelz study runs without errors"""
        self.run_optimizer(check_result=False, check_iter=False)

class TestBromineStudy(ForceBalanceSystemTest):

    def setup_method(self, method):
        super(TestBromineStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '003_liquid_bromine'))
        self.input_file='optimize.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_BROMINE_RESULTS"
        self.expected_results = EXPECTED_BROMINE_RESULTS
        self.absolute_tolerance = 0.10

    def test_bromine_study(self):
        """Check liquid bromine study converges to expected results"""
        self.run_optimizer()

class TestThermoBromineStudy(ForceBalanceSystemTest):
    def setup_method(self, method):
        super(TestThermoBromineStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '../../studies/004_thermo_liquid_bromine'))
        self.input_file='optimize.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_BROMINE_RESULTS"
        self.expected_results = EXPECTED_BROMINE_RESULTS
        self.absolute_tolerance = 0.05

    def test_thermo_bromine_study(self):
        """Check liquid bromine study (Thermo target) converges to expected results"""
        self.run_optimizer()


class EvaluatorServerMixin:
    """Mixin that manages an openff-evaluator server subprocess for tests.

    Subclasses should call ``_start_evaluator_server()`` in ``setup_method``
    and may set ``_cleanup_folders`` to a tuple of directory names to remove
    in ``teardown_method``.

    Notes
    -----
    - Output is redirected to a log file (not PIPE) so Dask worker subprocesses
      that inherit the fd don't fill the pipe buffer and deadlock.
    - ``python -u`` is used so each log line is flushed immediately to disk.
    - We poll the log file rather than making a raw TCP probe: a bare
      connect+disconnect causes recvall() in _handle_stream to return None,
      which crashes struct.unpack and kills the _handle_connections loop.
    """

    _server_port = 8000
    _cleanup_folders = ()

    def _start_evaluator_server(self):
        import subprocess, time
        self._server_log_path = os.path.abspath("server.log")
        self._server_log = open(self._server_log_path, "w")
        self.estimator_process = subprocess.Popen(
            ["python", "-u", "run_server.py", "-ngpus=0", "-ncpus=1"],
            stdout=self._server_log, stderr=self._server_log,
        )
        ready_marker = "listening at port {}".format(self._server_port)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.estimator_process.poll() is not None:
                self._server_log.flush()
                pytest.fail(
                    "Evaluator server process exited prematurely (rc=%d). Log:\n%s"
                    % (self.estimator_process.returncode, open(self._server_log_path).read()[-2000:])
                )
            with open(self._server_log_path) as f:
                if ready_marker in f.read():
                    break
            time.sleep(0.5)
        else:
            self.estimator_process.terminate()
            pytest.fail(
                "Evaluator server did not start within 120 seconds. Log:\n%s"
                % open(self._server_log_path).read()[-2000:]
            )

    def teardown_method(self):
        try:
            if hasattr(self, 'estimator_process') and self.estimator_process is not None:
                self.estimator_process.terminate()
                self.estimator_process.wait(timeout=10)
        except Exception:
            pass
        try:
            if hasattr(self, '_server_log') and self._server_log is not None:
                self._server_log.close()
            if hasattr(self, '_server_log_path') and os.path.exists(self._server_log_path):
                os.remove(self._server_log_path)
        except Exception:
            pass
        try:
            if hasattr(self, 'study_directory'):
                os.chdir(self.study_directory)
            for folder in self._cleanup_folders:
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
        finally:
            super().teardown_method()


@skip_openff_py39
class TestEvaluatorBromineStudy(EvaluatorServerMixin, ForceBalanceSystemTest):
    _cleanup_folders = ("working_directory", "stored_data")

    def setup_method(self, method):
        pytest.importorskip("openff.evaluator")
        super().setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '003d_evaluator_liquid_bromine'))
        self.study_directory = os.getcwd()
        targets = tarfile.open('targets.tar.gz', 'r')
        targets.extractall()
        targets.close()
        self._start_evaluator_server()
        self.input_file = 'gradient.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)

    def test_bromine_study(self):
        """Check bromine study produces objective function and gradient in expected range """
        objective = self.get_objective()
        try:
            data = objective.Full(np.zeros(objective.FF.np), 1, verbose=True)
        except Exception as exc:
            self._server_log.flush()
            raise RuntimeError(
                "objective.Full raised %s: %s\nServer log (last 2000 chars):\n%s"
                % (type(exc).__name__, exc, open(self._server_log_path).read()[-2000:])
            ) from exc
        X, G, H = data['X'], data['G'], data['H']
        np.testing.assert_allclose(EXPECTED_EVALUATOR_BROMINE_OBJECTIVE, X, atol=200,
            err_msg="\nObjective outside expected range. Update EXPECTED_EVALUATOR_BROMINE_OBJECTIVE if reasonable.")
        np.testing.assert_allclose(EXPECTED_EVALUATOR_BROMINE_GRADIENT, G, atol=4300,
            err_msg="\nGradient outside expected range. Update EXPECTED_EVALUATOR_BROMINE_GRADIENT if reasonable.")

class TestLipidStudy(ForceBalanceSystemTest):
    def setup_method(self, method):
        super(TestLipidStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '../../studies/010_lipid_study'))
        self.input_file='simple.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_LIPID_RESULTS"
        self.expected_results = EXPECTED_LIPID_RESULTS
        self.absolute_tolerance = 0.100

    def test_lipid_study(self):
        """Check lipid tutorial study runs without errors"""
        self.run_optimizer()

class TestImplicitSolventHFEStudy(ForceBalanceSystemTest):
    def setup_method(self, method):
        if not check_for_openmm(): pytest.skip("No OpenMM modules found.")
        super(TestImplicitSolventHFEStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '012_implicit_solvent_hfe'))
        self.input_file='optimize.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_ETHANOL_RESULTS"
        self.expected_results = EXPECTED_ETHANOL_RESULTS
        self.absolute_tolerance = 0.020
 
    def test_implicit_solvent_hfe_study(self):
        """Check implicit hydration free energy study (Hydration target) converges to expected results"""
        self.run_optimizer(check_result=False, check_iter=False, use_pvals=True)

@skip_openff_py39
class TestOpenFFTorsionProfileStudy(ForceBalanceSystemTest):
    def setup_method(self, method):
        pytest.importorskip("openff.toolkit", minversion="0.4")
        pytest.importorskip("openeye.oechem")
        super(TestOpenFFTorsionProfileStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '023_torsion_relaxed'))
        targets = tarfile.open('targets.tar.gz','r')
        targets.extractall()
        targets.close()
        self.input_file='optimize_minimal.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_OPENFF_TORSIONPROFILE_RESULTS"
        self.expected_results = EXPECTED_OPENFF_TORSIONPROFILE_RESULTS
        self.absolute_tolerance = 0.001

    def test_openff_torsionprofile_study(self):
        """Check OpenFF torsion profile optimization converges to expected results"""
        self.run_optimizer(check_iter=False)

@skip_openff_py39
class TestRechargeMethaneStudy(ForceBalanceSystemTest):

    def setup_method(self, method):

        pytest.importorskip("openff.recharge")

        super(TestRechargeMethaneStudy, self).setup_method(method)

        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '025_openff_recharge'))

        ## Extract targets archive.
        targets = tarfile.open('targets.tar.gz','r')
        targets.extractall()
        targets.close()

        self.input_file='optimize.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)

    def test_study(self):

        objective = self.get_objective()
        data      = objective.Full(np.zeros(objective.FF.np),1,verbose=True)
        X, G, H   = data['X'], data['G'], data['H']

        msgX=(
            "\nCalculated objective function is outside expected range.\n "
            "If this seems reasonable, update EXPECTED_RECHARGE_METHANE_ESP_OBJECTIVE "
            "and EXPECTED_RECHARGE_METHANE_FIELD_OBJECTIVE in test_system.py with "
            "these values"
        )
        np.testing.assert_allclose(
            (
                EXPECTED_RECHARGE_METHANE_ESP_OBJECTIVE
                + EXPECTED_RECHARGE_METHANE_FIELD_OBJECTIVE
            ),
            X,
            rtol=5.0e-7,
            err_msg=msgX
        )
        msgG = (
            "\nCalculated gradient is outside expected range.\n "
            "If this seems reasonable, update EXPECTED_RECHARGE_METHANE_ESP_GRADIENT "
            "and EXPECTED_RECHARGE_METHANE_FIELD_GRADIENT in test_system.py with "
            "these values"
        )
        np.testing.assert_allclose(
            (
                EXPECTED_RECHARGE_METHANE_ESP_GRADIENT
                + EXPECTED_RECHARGE_METHANE_FIELD_GRADIENT
            ),
            G,
            rtol=5.0e-7,
            err_msg=msgG
        )

@skip_openff_py39
class TestEvaluatorWaterVSiteStudy(EvaluatorServerMixin, ForceBalanceSystemTest):
    """Check that we can co-optimize pure and mixture properties of water with a virtual site.

    Physical values may be imprecise due to short simulations; this primarily
    checks that nothing breaks technically.
    """

    def setup_method(self, method):
        pytest.importorskip("openff.evaluator")
        pytest.importorskip("openff.toolkit")
        super().setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '028_smirnoff_tip4p_geometry_fit'))
        self.study_directory = os.getcwd()
        self._start_evaluator_server()
        self.input_file = 'optimize.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)
        self.expected_results_name = "EXPECTED_VSITE_VDW_PARAMETERS"
        self.expected_results = EXPECTED_VSITE_VDW_PARAMETERS
        self.absolute_tolerance = 0.02

    def test_water_vsite_study(self):
        """Check water virtual site study produces objective function and gradient in expected range"""
        self.run_optimizer(check_result=False, use_pvals=True)


@skip_openff_py39
class TestWaterVSiteGradients(ForceBalanceSystemTest):
    """Test that AbInitio_SMIRNOFF and AbInitio_OpenMM give identical
    objective values for TIP4P-FB and TIP5P water at mvals=0.
    
    This basically tests the Toolkit and Interchange parsing of virtual sites.
    """

    def setup_method(self, method):
        pytest.importorskip("openff.toolkit")
        super(TestWaterVSiteGradients, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies',
                              '029_smirnoff_vs_openmm_water_vsite'))
        self.study_directory = os.getcwd()
        self.atol = 1e-6

    def teardown_method(self):
        os.chdir(self.start_directory)

    def _eval(self, input_file):
        """Parse input_file, build objective, evaluate at mvals=0, return (X, G)."""
        options, tgt_opts = parse_inputs(input_file)
        ff  = FF(options)
        obj = Objective(options, tgt_opts, ff)
        data = obj.Full(np.zeros(ff.np), Order=1, verbose=True)
        return data['X'], data['G']

    def test_tip4p_smirnoff(self):
        """AbInitio_SMIRNOFF objective for TIP4P-FB is in expected range."""
        X, G = self._eval('gradient_tip4p_smirnoff.in')
        np.testing.assert_allclose(
            EXPECTED_TIP4P_OBJECTIVE, X, atol=self.atol,
            err_msg="TIP4P SMIRNOFF objective changed; update EXPECTED_TIP4P_OBJECTIVE"
        )

    def test_tip4p_openmm(self):
        """AbInitio_OpenMM objective for TIP4P-FB matches SMIRNOFF value."""
        X, G = self._eval('gradient_tip4p_openmm.in')
        np.testing.assert_allclose(
            EXPECTED_TIP4P_OBJECTIVE, X, atol=self.atol,
            err_msg="TIP4P OpenMM objective changed; update EXPECTED_TIP4P_OBJECTIVE"
        )

    def test_tip5p_smirnoff(self):
        """AbInitio_SMIRNOFF objective for TIP5P is in expected range."""
        X, G = self._eval('gradient_tip5p_smirnoff.in')
        np.testing.assert_allclose(
            EXPECTED_TIP5P_OBJECTIVE, X, atol=self.atol,
            err_msg="TIP5P SMIRNOFF objective changed; update EXPECTED_TIP5P_OBJECTIVE"
        )

    def test_tip5p_openmm(self):
        """AbInitio_OpenMM objective for TIP5P matches SMIRNOFF value."""
        X, G = self._eval('gradient_tip5p_openmm.in')
        np.testing.assert_allclose(
            EXPECTED_TIP5P_OBJECTIVE, X, atol=self.atol,
            err_msg="TIP5P OpenMM objective changed; update EXPECTED_TIP5P_OBJECTIVE"
        )