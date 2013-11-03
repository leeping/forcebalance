import unittest
import os, sys
import tarfile
from __init__ import ForceBalanceTestCase
from forcebalance.nifty import printcool_dictionary
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer, Counter
from collections import OrderedDict
from numpy import array

# expected results taken from previous runs. Update this if it changes and seems reasonable (updated 10/24/13)
EXPECTED_WATER_RESULTS = array([3.3192e-02, 4.3287e-02, 5.5072e-03, -4.5933e-02, 1.5499e-02, -3.7655e-01, 2.4720e-03, 1.1914e-02, 1.5066e-01])

# expected results taken from previous runs. Update this if it changes and seems reasonable (updated 11/02/13)
EXPECTED_BROMINE_RESULTS = array([-2.3426e-01, -3.1603e-02])

# fail test if we take more than this many iterations to converge. Update this as necessary
ITERATIONS_TO_CONVERGE = 5

class TestWaterTutorial(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/001_water_tutorial')
        targets = tarfile.open('targets.tar.bz2','r')
        targets.extractall()
        targets.close()

    def tearDown(self):
        os.system('rm -rf results targets backups temp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check water tutorial study runs without errors"""
        self.logger.debug("\nSetting input file to 'very_simple.in'\n")
        input_file='very_simple.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        forcefield  = FF(options)
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")
        self.logger.debug(str(optimizer) + "\n")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()
        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

        self.assertNdArrayEqual(EXPECTED_WATER_RESULTS,result,delta=0.001,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_WATER_RESULTS in test_system.py with these values")

        # Fail if calculation takes longer than previously to converge
        self.assertGreaterEqual(ITERATIONS_TO_CONVERGE, Counter(), msg="\nCalculation took longer than expected to converge (%d iterations vs previous of %d)" %\
        (ITERATIONS_TO_CONVERGE, Counter()))

class TestVoelzStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/009_voelz_nspe')

    def tearDown(self):
        os.system('rm -rf results backups temp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check voelz study runs without errors"""
        self.logger.debug("\nSetting input file to 'options.in'\n")
        input_file='options.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        self.logger.debug("Creating forcefield using loaded options: ")
        forcefield  = FF(options)
        self.logger.debug(str(forcefield) + "\n")
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        self.logger.debug("Creating object using loaded options and forcefield: ")
        objective   = Objective(options, tgt_opts, forcefield)
        self.logger.debug(str(objective) + "\n")
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.logger.debug(str(optimizer) + "\n")
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()

        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

class TestBromineStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/003_liquid_bromine')

    def tearDown(self):
        os.system('rm -rf results backups temp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check liquid bromine study converges to expected results"""
        self.logger.debug("\nSetting input file to 'options.in'\n")
        input_file='optimize.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        self.logger.debug("Creating forcefield using loaded options: ")
        forcefield  = FF(options)
        self.logger.debug(str(forcefield) + "\n")
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        self.logger.debug("Creating object using loaded options and forcefield: ")
        objective   = Objective(options, tgt_opts, forcefield)
        self.logger.debug(str(objective) + "\n")
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.logger.debug(str(optimizer) + "\n")
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()

        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

        self.assertNdArrayEqual(EXPECTED_BROMINE_RESULTS,result,delta=0.01,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_BROMINE_RESULTS in test_system.py with these values")

if __name__ == '__main__':
    unittest.main()
