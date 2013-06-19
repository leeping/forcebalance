import unittest, os, re
import forcebalance
import __init__

EXCLUDE = ['__init__.py', '__main__.py']   # list of modules to exclude from testing
TEST_MODULES=[module[:-3] for module in os.listdir('test')
                     if re.compile(".*\.py$").match(module)
                     and module not in EXCLUDE]

def runAllTests():
    print "\x1b[2J\x1b[80A"

    allTests = unittest.TestSuite()

    for module in TEST_MODULES:
        try:
            m=__import__(module)
            module_tests=unittest.defaultTestLoader.loadTestsFromModule(m)
            allTests.addTest(module_tests)
        except: pass
    runner=__init__.ForceBalanceTestRunner()

    allResults=runner.run(allTests)
    print allResults
    return allResults

#### main block ####
runAllTests()
    