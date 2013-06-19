import unittest
import os, sys
import traceback

class ForceBalanceTestCase(unittest.TestCase):
    def shortDescription(self):
        """Default shortDescription function returns None value if no description
        is present, but this causes errors when trying to print. Return empty
        string instead"""

        message = super(ForceBalanceTestCase,self).shortDescription()
        if message: return message
        else: return ""

class ForceBalanceTestResult(unittest.TestResult):
    """This manages the reporting of test results as they are run,
       and also records results in the internal data structures provided
       by unittest.TestResult"""

    def startTest(self, test):
        super(ForceBalanceTestResult, self).startTest(test)
        sys.stderr.write("...     " + test.id())

    def addFailure(self, test, err):
        super(ForceBalanceTestResult, self).addFailure(test,err)
        sys.stderr.write("\r\x1b[31;1m" + "FAIL" + "\x1b[0m    " + test.id() + "\n")
        
        errorMessage = self.buildErrorMessage(test, err)

        for line in errorMessage.splitlines():
            sys.stderr.write("\t >\t" + line + "\n")

    def addError(self, test, err):
        super(ForceBalanceTestResult, self).addError(test,err)
        sys.stderr.write("\r\x1b[33;1mERROR\x1b[0m   " + test.id() + "\n")

        errorMessage = self.buildErrorMessage(test,err)

        for line in errorMessage.splitlines():
            sys.stderr.write("\t >\t" + line + "\n")
    
    def buildErrorMessage(self, test, err):
        """Compile error data from test exceptions into a helpful message"""
        errorMessage = ""

        errorMessage += "Description:\n"
        if test.shortDescription(): errorMessage+=test.shortDescription()
        else: errorMessage+="No description available"
        errorMessage += "\n\n"

        errorMessage += traceback.format_exc() + "\n"
        return errorMessage

    def addSuccess(self, test):
        sys.stderr.write("\r\x1b[32mOK\x1b[0m      " + test.id() + "\n")

    def stopTestRun(self, test):
        super(ForceBalanceTestResult, self).stopTestRun(test)
        sys.stderr.write("\nAll tests successful\n")
        

class ForceBalanceTestRunner(object):
    """This test runner class manages the running and logging of tests.
       It controls WHERE test results go but not what is recorded.
       Once the tests have finished running, it will return the test result
       for further analysis"""

    def run(self, suite, verbose=False, stdout=os.devnull):
        if not verbose:        
            self.console = sys.stdout
            sys.stdout = open(stdout, 'w')

        result = ForceBalanceTestResult()
        suite.run(result)

        if not verbose:
            sys.stdout.close()
            sys.stdout = self.console

        return result
        
