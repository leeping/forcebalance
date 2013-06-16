import unittest, os, re
import forcebalance

EXCLUDE = ['__init__.py']   # list of modules to exclude from testing
TEST_MODULES=[module[:-3] for module in os.listdir(forcebalance.__path__[0])
                     if re.compile(".*\.py$").match(module)
                     and module not in EXCLUDE]

class ForceBalanceTestResult(unittest.TestResult):
    def __init__(self):
        super(ForceBalanceTestResult,self).__init__()
        self.failed=[]

    def startTestRun(self, test):
        """Run before first test"""
        pass

    def stopTestRun(self, test):
        """Run after last test"""
        pass

    def startTest(self, test):
        """Run before TestCase object 'test'"""
        pass

    def stopTest(self, test):
        """Run after TestCase object 'test'"""
        for failure in self.failures:
            if test==failure[0]: self.failed.append(test)   #somewhat redundant, in future use self.failures

class ForceBalanceTestRunner():
    def __init__(self, logfile=os.devnull):
        self.results=ForceBalanceTestResult()
        self.log = open(logfile,'w')

    def run(self, suites):
        for module in suites:
            for test in module:
                description = test.shortDescription()
                if description is None: description = test.id()
                print "\t" + description,
                test.run(self.results)
                if test in self.results.failed:
                    print "\r\x1b[31m" + "FAIL" + "\x1b[0m\t" + description
                    self.log.write("FAIL\t" + description + "\n")
                else:
                    print "\r\x1b[32m" + "OK" + "\x1b[0m\t" + description
                    self.log.write("OK\t" + description + "\n")
        return self.results

def runAllTests():
    print "\x1b[2J\x1b[80A"

    runner=ForceBalanceTestRunner()

    # currently runs each module with separate test runner- should run one test runner for test suite containing all modules
    for module in TEST_MODULES:
        try:
            m=__import__(module)
            module_tests=unittest.defaultTestLoader.loadTestsFromModule(m)
            print "Loaded %d tests for module 'forcebalance.%s' :" % (module_tests.countTestCases(),module)
            runner.run(module_tests)
        except: pass    # todo: should we do anything for modules without tests
            

runAllTests()
    