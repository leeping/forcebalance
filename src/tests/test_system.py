from __future__ import absolute_import

from builtins import str
import os, shutil
import tarfile
from .__init__ import ForceBalanceTestCase, check_for_openmm
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer, Counter
from numpy import array
import numpy as np
import pytest

# expected results (mvals) taken from previous runs. Update this if it changes and seems reasonable (updated 10/24/13)
#EXPECTED_WATER_RESULTS = array([3.3192e-02, 4.3287e-02, 5.5072e-03, -4.5933e-02, 1.5499e-02, -3.7655e-01, 2.4720e-03, 1.1914e-02, 1.5066e-01])
EXPECTED_WATER_RESULTS = array([4.2370e-02, 3.1217e-02, 5.6925e-03, -4.8114e-02, 1.6735e-02, -4.1722e-01, 6.2716e-03, 4.6306e-03, 2.5960e-01])

# expected results (mvals) taken from previous runs. Update this if it changes and seems reasonable (updated 01/24/14)
EXPECTED_BROMINE_RESULTS = array([-0.305718, -0.12497])

# expected objective function from 003d evaluator bromine study. (updated 11/23/19)
EXPECTED_EVALUATOR_BROMINE_OBJECTIVE = array([1000])

# expected gradient elements from 003d evaluator bromine study. Very large uncertainties of +/- 2000 (updated 11/23/19)
EXPECTED_EVALUATOR_BROMINE_GRADIENT = array([4500, 5500])

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

class TestEvaluatorBromineStudy(ForceBalanceSystemTest):
    def setup_method(self, method):
        pytest.importorskip("openff.evaluator")
        super(TestEvaluatorBromineStudy, self).setup_method(method)
        cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cwd, '..', '..', 'studies', '003d_evaluator_liquid_bromine'))
        ## Extract targets archive.
        targets = tarfile.open('targets.tar.gz','r')
        targets.extractall()
        targets.close()
        ## Start the estimator server.
        import subprocess, time
        self.estimator_process = subprocess.Popen([
            "python", "run_server.py", "-ngpus=0", "-ncpus=1"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ## Give the server time to start.
        time.sleep(5)
        self.input_file='gradient.in'
        self.logger.debug("\nSetting input file to '%s'\n" % self.input_file)

    def teardown_method(self):
        self.estimator_process.terminate()
        shutil.rmtree("working_directory")
        shutil.rmtree("stored_data")
        super(TestEvaluatorBromineStudy, self).teardown_method()

    def test_bromine_study(self):
        """Check bromine study produces objective function and gradient in expected range """
        objective = self.get_objective()
        data      = objective.Full(np.zeros(objective.FF.np),1,verbose=True)
        X, G, H   = data['X'], data['G'], data['H']
        msgX="\nCalculated objective function is outside expected range.\n If this seems reasonable, update EXPECTED_EVALUATOR_BROMINE_OBJECTIVE in test_system.py with these values"
        np.testing.assert_allclose(EXPECTED_EVALUATOR_BROMINE_OBJECTIVE, X, atol=200, err_msg=msgX)
        msgG="\nCalculated gradient is outside expected range.\n If this seems reasonable, update EXPECTED_EVALUATOR_BROMINE_GRADIENT in test_system.py with these values"
        np.testing.assert_allclose(EXPECTED_EVALUATOR_BROMINE_GRADIENT, G, atol=4000, err_msg=msgG)

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
