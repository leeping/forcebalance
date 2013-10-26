import unittest
import os, sys, time, re
import traceback
from collections import OrderedDict
import numpy
import forcebalance.output

forcebalance.output.getLogger("forcebalance.test").propagate=False

os.chdir(os.path.dirname(__file__) + "/..")
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

        self.logger = forcebalance.output.getLogger('forcebalance.test.' + __name__[5:])

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
            reason += '\n'
            if self.longMessage and msg:
                reason += msg
            raise self.failureException(reason)
            
    def assertEqual(self, first, second, msg=None):
        self.logger.debug(">ASSERT(%s==%s)\n" % (str(first), str(second)))
        return super(ForceBalanceTestCase,self).assertEqual(first,second,msg)
        
    def assertNotEqual(self, first, second, msg=None):
        self.logger.debug(">ASSERT(%s!=%s)\n" % (str(first), str(second)))
        return super(ForceBalanceTestCase,self).assertNotEqual(first,second,msg)
        
    def assertTrue(self, expr, msg=None):
        self.logger.debug(">ASSERT(%s)\n" % (str(expr)))
        return super(ForceBalanceTestCase,self).assertTrue(expr, msg)
        
class ForceBalanceTestResult(unittest.TestResult):
    """This manages the reporting of test results as they are run,
       and also records results in the internal data structures provided
       by unittest.TestResult"""

    def __init__(self):
        """Add logging capabilities to the standard TestResult implementation"""
        super(ForceBalanceTestResult,self).__init__()
        self.logger = forcebalance.output.getLogger('forcebalance.test.results')

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
        self.logger.debug("\nBeginning ForceBalance test suite at %s\n" % time.strftime('%x %X %Z'))

    def stopTestRun(self, test):
        """Run after all tests have finished"""

        self.runTime = time.time()-self.runTime
        self.logger.debug("\nCompleted test suite\n")
        self.logger.info("\n<run=%d errors=%d fail=%d in %.2fs>\n" % (self.testsRun,len(self.errors),len(self.failures), self.runTime))
        if self.wasSuccessful(): self.logger.info("All tests passed successfully\n")
        else: 
            self.logger.info("Some tests failed or had errors!\n")
            sys.exit(1)

class ForceBalanceTestRunner(object):
    """This test runner class manages the running and logging of tests.
       It controls WHERE test results go but not what is recorded.
       Once the tests have finished running, it will return the test result
       in the standard unittest.TestResult format"""
    def __init__(self, logger=forcebalance.output.getLogger("forcebalance.test"), verbose = False):
        self.logger = logger
        forcebalance.output.getLogger("forcebalance.test")
        
    def check(self, test_modules=__all__):
        """This tries importing test modules which is helpful for error checking
        since the unittest loader is not very good at identifying syntax errors
        when discovering tests. Checking that test_modules are all importable
        produced better, more informative exceptions and lets you know when your
        test modules have syntax errors"""
        
        # if test suite is being running from within forcebalance module, append the forcebalance prefix
        if __name__=="forcebalance.test.__init__":
            test_modules = ["forcebalance.test." + test_module for test_module in test_modules]
            
        for test_module in test_modules:
                __import__(test_module)


    def run(self,
            test_modules=__all__,
            pretend=False,
            logfile='test/test.log',
            loglevel=forcebalance.output.INFO,
            **kwargs):
            
        self.check()

        self.logger.setLevel(loglevel)

        # first install unittest interrupt handler which gracefully finishes current test on Ctrl+C
        unittest.installHandler()

        # create blank test suite and fill it with test suites loaded from each test module
        tests = unittest.TestSuite()
        systemTests = unittest.TestSuite()
        for suite in unittest.defaultTestLoader.discover('test'):
            for module in suite:
                for test in module:
                    modName,caseName,testName = test.id().split('.')
                    if modName in test_modules:
                        if modName=="test_system": systemTests.addTest(test)
                        else: tests.addTest(test)

        tests.addTests(systemTests) # integration tests should be run after other tests

        result = ForceBalanceTestResult()
        
        forcebalance.output.getLogger("forcebalance").addHandler(forcebalance.output.NullHandler())

        ### START TESTING ###
        # run any pretest tasks before first test
        result.startTestRun(tests)

        # if pretend option is enabled, skip all tests instead of running them
        if pretend:
            for test in tests:
                result.addSkip(test)

        # otherwise do a normal test run
        else:
            unittest.registerResult(result)
            try:
                tests.run(result)
            except KeyboardInterrupt:
                # Adding this allows us to determine
                # what is causing tests to hang indefinitely.
                import traceback
                traceback.print_exc()
                self.logger.exception(msg="Test run cancelled by user")
            except:
                self.logger.exception(msg="An unexpected exception occurred while running tests\n")

        result.stopTestRun(tests)
        ### STOP TESTING ###

        return result
