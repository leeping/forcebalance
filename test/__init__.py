import unittest
import os, sys, time
import traceback
from collections import OrderedDict
import numpy
import forcebalance

class ForceBalanceTestCase(unittest.TestCase):
    def __init__(self,methodName='runTest'):
        """Override default test case constructor to set longMessage=True, reset cwd after test
        @override unittest.TestCase.__init(methodName='runTest')"""

        super(ForceBalanceTestCase,self).__init__(methodName)
        self.longMessage=True
        self.addCleanup(os.chdir, os.getcwd())  # directory changes shouldn't persist between tests
        self.addTypeEqualityFunc(numpy.ndarray, self.assertNdArrayEqual)

        self.logger = forcebalance.logging.getLogger('test.' + __name__[5:])

    def shortDescription(self):
        """Default shortDescription function returns None value if no description
        is present, but this causes errors when trying to print. Return empty string instead
        @override unittest.TestCase.shortDescription()"""

        message = super(ForceBalanceTestCase,self).shortDescription()
        if message: return message
        else: return self.id()
    
    def assertNdArrayEqual(self, A, B, msg=None, delta=.00001):
        """Provide equality checking for numpy arrays, with informative error messages
        when applicable. A and B are equal if they have the same dimensions and
        for all elements a in A and corresponding elements b in B,
        a == b +/- delta"""
        
        if A.shape != B.shape:
            reason = "Tried to compare ndarray of size %s to ndarray of size %s\n" % (str(A.shape),str(B.shape))
            if self.longMessage and msg:
                reason += msg
            raise self.failureException(reason)

        unequal = (abs(A-B)>delta)
        if unequal.any():
            reason = "ndarrays not equal"
            indexes = numpy.argwhere(unequal)
            n = len(indexes.tolist())
            for j, index in enumerate(numpy.argwhere(unequal)):
                # try printing first and last few unequal values
                if j>=4 and n>9 and n-j>4:
                    if j==4: reason += "\n[...]"
                    continue
                else: reason += "\nA[%s]\t%s =! %s\tB[%s]" % (index[0],A[index[0]],B[index[0]],index[0])
            if self.longMessage and msg:
                reason += msg
            raise self.failureException(reason)

class ForceBalanceTestResult(unittest.TestResult):
    """This manages the reporting of test results as they are run,
       and also records results in the internal data structures provided
       by unittest.TestResult"""

    def __init__(self):
        """Add logging capabilities to the standard TestResult implementation"""
        super(ForceBalanceTestResult,self).__init__()
        self.logger = forcebalance.logging.getLogger('test.results')

    def startTest(self, test):
        """Notify of test start by writing message to stderr, and also printing to stdout
        @override unittest.TestResult.startTest(test)"""

        super(ForceBalanceTestResult, self).startTest(test)
        self.logger.info("---     " + test.shortDescription())
        print "<<<<<<<< Starting %s >>>>>>>>\n" % test.id()

    def addFailure(self, test, err):
        """Run whenever a test comes back as failed
        @override unittest.TestResult.addFailure(test,err)"""

        super(ForceBalanceTestResult, self).addFailure(test,err)
        self.logger.info("\r\x1b[31;1m" + "FAIL" + "\x1b[0m    " + test.shortDescription() + "\n")
        
        errorMessage = self.buildErrorMessage(test, err)

        for line in errorMessage.splitlines():
            self.logger.info("\t >\t" + line + "\n")

    def addError(self, test, err):
        """Run whenever a test comes back with an unexpected exception
        @override unittest.TestResult.addError(test,err)"""

        super(ForceBalanceTestResult, self).addError(test,err)
        self.logger.info("\r\x1b[33;1mERROR\x1b[0m   " + test.shortDescription() + "\n")

        errorMessage = self.buildErrorMessage(test,err)

        for line in errorMessage.splitlines():
            self.logger.info("\t >\t" + line + "\n")
    
    def buildErrorMessage(self, test, err):
        """Compile error data from test exceptions into a helpful message"""

        errorMessage = ""
        errorMessage += test.id()
        errorMessage += "\n\n"

        errorMessage += traceback.format_exc() + "\n"
        return errorMessage

    def addSuccess(self, test):
        """Run whenever a test comes back as passed
        @override unittest.TestResult.addSuccess(test)"""

        self.logger.info("\r\x1b[32mOK\x1b[0m      " + test.shortDescription() + "\n")

    def addSkip(self, test, err=""):
        """Run whenever a test is skipped
        @override unittest.TestResult.addSkip(test,err)"""

        self.logger.info("\r\x1b[33;1mSKIP\x1b[0m    " + test.shortDescription() + "\n")
        if err: self.logger.info("\t\t%s\n" % err)

    def stopTest(self, test):
        """Run whenever a test is finished, regardless of the result
        @override unittest.TestResult.stopTest(test)"""
        print "\n<<<<<<<< Finished %s >>>>>>>>\n\n" % test.id()

    def startTestRun(self, test):
        """Run before any tests are started"""
        self.runTime= time.time()

    def stopTestRun(self, test):
        """Run after all tests have finished"""

        self.runTime = time.time()-self.runTime
        self.logger.info("\n<run=%d errors=%d fail=%d in %.2fs>\n" % (self.testsRun,len(self.errors),len(self.failures), self.runTime))
        if self.wasSuccessful(): self.logger.info("All tests passed successfully\n")
        else: self.logger.info("Some tests failed or had errors!\n")

class ForceBalanceTestRunner(object):
    """This test runner class manages the running and logging of tests.
       It controls WHERE test results go but not what is recorded.
       Once the tests have finished running, it will return the test result
       in the standard unittest.TestResult format"""
    def __init__(self, logger=forcebalance.logging.getLogger("test"), verbose = False):
        self.logger = logger

    def run(self,test_modules=[],pretend=False,program_output='test/test.log',quick=False, verbose=False):
        if verbose:
            self.logger.setLevel(forcebalance.logging.DEBUG)
        else:
            self.logger.setLevel(forcebalance.logging.INFO)

        # first install unittest interrupt handler which gracefully finishes current test on Ctrl+C
        unittest.installHandler()

        # create blank test suite and fill it with test suites loaded from each test module
        tests = unittest.TestSuite()
        for module in test_modules:
            try:
                m=__import__(module)
                module_tests=unittest.defaultTestLoader.loadTestsFromModule(m)
                tests.addTest(module_tests)
            except:
                self.logger.critical("Error loading '%s'\n" % module)
                print traceback.print_exc()

        result = ForceBalanceTestResult()

        ### START TESTING ###
        # run any pretest tasks before first test
        result.startTestRun(tests)

        # if pretend option is enabled, skip all tests instead of running them
        if pretend:
            for module in tests:
                for test in module:
                    try:
                        result.addSkip(test)
                    # addSkip will fail if run on TestSuite objects
                    except AttributeError: continue

        # otherwise do a normal test run
        else:
            self.console = sys.stdout
            sys.stdout = open(program_output, 'w')

            self.logger.addHandler(forcebalance.nifty.RawStreamHandler(sys.stderr))
            self.logger.setLevel(forcebalance.logging.DEBUG)

            unittest.registerResult(result)
            tests.run(result)

            sys.stdout.close()
            sys.stdout = self.console

        result.stopTestRun(tests)
        ### STOP TESTING ###

        return result
        
class TestValues(object):
    """Contains values used as inputs, defaults, etc during testing"""
    opts={
            'penalty_type': 'L2',
            'print_gradient': 1, 
            'eig_lowerbound': 0.0001, 
            'error_tolerance': 0.0, 
            'scanindex_name': [], 
            'read_mvals': None, 
            'maxstep': 100, 
            'print_parameters': 1, 
            'penalty_hyperbolic_b': 1e-06, 
            'gmxsuffix': '', 
            'readchk': None, 
            'mintrust': 0.0, 
            'penalty_multiplicative': 0.0, 
            'convergence_step': 0.0001, 
            'adaptive_damping': 0.5, 
            'finite_difference_h': 0.001, 
            'wq_port': 0, 
            'verbose_options': 0,
            'scan_vals': None,
            'logarithmic_map': 0,
            'writechk_step': 1,
            'forcefield': ['water.itp'],
            'use_pvals': 0,
            'scanindex_num': [],
            'normalize_weights': 1,
            'adaptive_factor': 0.25,
            'trust0': 0.1,
            'gmxpath': '/usr/bin',
            'writechk': None,
            'print_hessian': 0,
            'have_vsite': 0,
            'tinkerpath': '',
            'ffdir': 'forcefield',
            'constrain_charge': 0,
            'convergence_gradient': 0.0001,
            'convergence_objective': 0.0001,
            'backup': 0,
            'rigid_water': 0,
            'search_tolerance': 0.0001,
            'objective_history': 2,
            'amoeba_polarization': 'direct',
            'lm_guess': 1.0,
            'priors': OrderedDict(),
            'asynchronous': 0,
            'read_pvals': None,
            'root': '.',
            'penalty_alpha': 0.001,
            'penalty_additive': 0.01}

    tgt_opt = {   
            'fdgrad': 0, 
            'qmboltz': 0.0, 
            'gas_prod_steps': 0, 
            'force': 1, 
            'weight': 1.0, 
            'fd_ptypes': [], 
            'resp': 0, 
            'sleepy': 0, 
            'fitatoms': 0, 
            'w_force': 1.0, 
            'w_cp': 1.0, 
            'batch_fd': 0, 
            'w_resp': 0.0, 
            'force_cuda': 0, 
            'w_netforce': 0.0, 
            'mts_vvvr': 0, 
            'do_cosmo': 0, 
            'quadrupole_denom': 1.0, 
            'self_pol_mu0': 0.0, 
            'qmboltztemp': 298.15, 
            'force_map': 'residue', 
            'self_pol_alpha': 0.0, 
            'wavenumber_tol': 10.0, 
            'energy_upper': 30.0, 
            'cauchy': 0, 
            'w_torque': 0.0, 
            'w_alpha': 1.0, 
            'w_eps0': 1.0, 
            'openmm_cuda_precision': '', 
            'gas_equ_steps': 0, 
            'masterfile': 'interactions.txt', 
            'absolute': 0, 
            'anisotropic_box': 0, 
            'fragment1': '', 
            'rmsd_denom': 0.1, 
            'dipole_denom': 1.0, 
            'fdhessdiag': 0, 
            'energy': 1, 
            'energy_denom': 1.0, 
            'covariance': 0, 
            'hvap_subaverage': 0, 
            'optimize_geometry': 1, 
            'liquid_interval': 0.05, 
            'w_energy': 1.0, 
            'w_rho': 1.0, 
            'liquid_equ_steps': 10000, 
            'fragment2': '', 
            'w_hvap': 1.0, 
            'w_kappa': 1.0, 
            'attenuate': 0, 
            'polarizability_denom': 1.0, 
            'manual': 0, 
            'run_internal': 1, 
            'sampcorr': 0, 
            'fdhess': 0, 
            'liquid_prod_steps': 20000, 
            'liquid_timestep': 0.5, 
            'shots': -1, 
            'resp_a': 0.001, 
            'resp_b': 0.1, 
            'whamboltz': 0, 
            'all_at_once': 1,
            'root':'.',
            'jobtype': 'NEWTON',
            'writelevel':0}

    # target options in 001_water_tutorial
    water_tgt_opts = [tgt_opt.copy(), tgt_opt.copy()]
    water_tgt_opts[0].update({'type': 'ABINITIO_GMX', 'name': 'cluster-06'})
    water_tgt_opts[1].update({'type': 'ABINITIO_GMX', 'name': 'cluster-12'})

