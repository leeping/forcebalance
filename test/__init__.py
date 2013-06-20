import unittest
import os, sys
import traceback

class ForceBalanceTestCase(unittest.TestCase):
    def __init__(self,methodName='runTest'):
        super(ForceBalanceTestCase,self).__init__(methodName)
        self.longMessage=True

    def shortDescription(self):
        """Default shortDescription function returns None value if no description
        is present, but this causes errors when trying to print. Return empty
        string instead"""

        message = super(ForceBalanceTestCase,self).shortDescription()
        if message: return message
        else: return self.id()

class ForceBalanceTestResult(unittest.TestResult):
    """This manages the reporting of test results as they are run,
       and also records results in the internal data structures provided
       by unittest.TestResult"""

    def startTest(self, test):
        super(ForceBalanceTestResult, self).startTest(test)
        sys.stderr.write("---     " + test.shortDescription())
        print "<<<<<<<< Starting test %s >>>>>>>>\n" % test.id()

    def addFailure(self, test, err):
        super(ForceBalanceTestResult, self).addFailure(test,err)
        sys.stderr.write("\r\x1b[31;1m" + "FAIL" + "\x1b[0m    " + test.shortDescription() + "\n")
        
        errorMessage = self.buildErrorMessage(test, err)

        for line in errorMessage.splitlines():
            sys.stderr.write("\t >\t" + line + "\n")

    def addError(self, test, err):
        super(ForceBalanceTestResult, self).addError(test,err)
        sys.stderr.write("\r\x1b[33;1mERROR\x1b[0m   " + test.shortDescription() + "\n")

        errorMessage = self.buildErrorMessage(test,err)

        for line in errorMessage.splitlines():
            sys.stderr.write("\t >\t" + line + "\n")
    
    def buildErrorMessage(self, test, err):
        """Compile error data from test exceptions into a helpful message"""
        errorMessage = ""
        errorMessage += test.id()
        errorMessage += "\n\n"

        errorMessage += traceback.format_exc() + "\n"
        return errorMessage

    def addSuccess(self, test):
        sys.stderr.write("\r\x1b[32mOK\x1b[0m      " + test.shortDescription() + "\n")

    def addSkip(self, test, err=""):
        sys.stderr.write("\r\x1b[33;1mSKIP\x1b[0m    " + test.shortDescription() + "\n")
        if err: sys.stderr.write("\t\t%s\n" % err)

    def stopTest(self, test):
        print "\n<<<<<<<< Finished test %s >>>>>>>>\n\n" % test.id()

class ForceBalanceTestRunner(object):
    """This test runner class manages the running and logging of tests.
       It controls WHERE test results go but not what is recorded.
       Once the tests have finished running, it will return the test result
       for further analysis"""

    def run(self,test_modules=[],exclude=[],pretend=False,program_output='test/test.log',quick=False):   
        unittest.installHandler()

        tests = unittest.TestSuite()
        for module in test_modules:
            try:
                m=__import__(module)
                module_tests=unittest.defaultTestLoader.loadTestsFromModule(m)
                tests.addTests(module_tests)
            except: print "Error loading '%s'" % module

        result = ForceBalanceTestResult()
        if pretend:
            for module in tests:
                for test in module:
                    try:
                        result.addSkip(test)
                    except AttributeError: continue
        else:
            self.console = sys.stdout
            sys.stdout = open(program_output, 'w')

            unittest.registerResult(result)
            tests.run(result)

            sys.stdout.close()
            sys.stdout = self.console

        return result
        
