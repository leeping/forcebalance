import unittest
import os, sys, time, re
import traceback
from collections import OrderedDict
import numpy
import forcebalance
from forcebalance import logging

__all__ = [module[:-3] for module in sorted(os.listdir('test'))
           if re.match("^test_.*\.py$",module)]

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
        self.logger.debug("\n>>>     Starting %s\n" % test.id())
        self.logger.info(">>>     " + test.shortDescription())

    def addFailure(self, test, err):
        """Run whenever a test comes back as failed
        @override unittest.TestResult.addFailure(test,err)"""

        super(ForceBalanceTestResult, self).addFailure(test,err)
        self.logger.warning("\r\x1b[31;1m" + "FAIL" + "\x1b[0m    " + test.shortDescription() + "\n")
        
        errorMessage = self.buildErrorMessage(test, err)

        for line in errorMessage.splitlines():
            self.logger.warning("\t >\t" + line + "\n")

    def addError(self, test, err):
        """Run whenever a test comes back with an unexpected exception
        @override unittest.TestResult.addError(test,err)"""

        super(ForceBalanceTestResult, self).addError(test,err)
        self.logger.warning("\r\x1b[33;1mERROR\x1b[0m   " + test.shortDescription() + "\n")

        errorMessage = self.buildErrorMessage(test,err)

        for line in errorMessage.splitlines():
            self.logger.warning("\t >\t" + line + "\n")
    
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
        self.logger.debug(">>>     Finished %s\n\n" % test.id())

    def startTestRun(self, test):
        """Run before any tests are started"""
        self.runTime= time.time()
        self.logger.debug("\nBeginning test suite\n")

    def stopTestRun(self, test):
        """Run after all tests have finished"""

        self.runTime = time.time()-self.runTime
        self.logger.debug("\nCompleted test suite\n")
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

    def run(self,
            test_modules=__all__,
            pretend=False,
            logfile='test/test.log',
            loglevel=logging.INFO,
            **kwargs):

        self.logger.setLevel(loglevel)

        # first install unittest interrupt handler which gracefully finishes current test on Ctrl+C
        unittest.installHandler()

        # create blank test suite and fill it with test suites loaded from each test module
        tests = unittest.TestSuite()
        for module in test_modules:
            try:
                m=__import__(module)
                module_tests=unittest.defaultTestLoader.loadTestsFromModule(m)
                tests.addTest(module_tests)
            except ImportError:
                self.logger.error("No such test module: %s\n" % module)
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
            sys.stdout = open(logfile, 'w')

            unittest.registerResult(result)
            tests.run(result)

            sys.stdout.close()
            sys.stdout = self.console

        result.stopTestRun(tests)
        ### STOP TESTING ###

        return result