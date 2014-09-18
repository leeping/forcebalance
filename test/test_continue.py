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

class TestWaterTutorial(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.system('rm -rf test/files/test_continue.run')
        os.makedirs('test/files/test_continue.run')
        os.system('cp -r studies/001_water_tutorial/forcefield test/files/test_continue.run/forcefield')
        os.system('cp -r studies/001_water_tutorial/targets test/files/test_continue.run/targets')
        os.chdir('test/files')
        os.system('cp -r test_continue.sav test_continue.in test_continue.tmp test_continue.run')
        os.chdir('test_continue.run')

    def tearDown(self):
        os.chdir('..')
        os.system('rm -rf test_continue.run')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check continuation from a previous run"""
        self.logger.debug("\nSetting input file to 'test_continue.in'\n")
        input_file='test_continue.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        options['continue'] = True
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

        self.assertEqual(optimizer.iterinit, 2, msg="\nInitial iteration counter is incorrect")
        self.assertEqual(optimizer.iteration, 2, msg="\nFinal iteration counter is incorrect")

        # self.assertNdArrayEqual(EXPECTED_WATER_RESULTS,result,delta=0.001,
        #                         msg="\nCalculation results have changed from previously calculated values.\n"
        #                         "If this seems reasonable, update EXPECTED_WATER_RESULTS in test_system.py with these values")
        # # Fail if calculation takes longer than previously to converge
        # self.assertGreaterEqual(ITERATIONS_TO_CONVERGE, Counter(), msg="\nCalculation took longer than expected to converge (%d iterations vs previous of %d)" %\
        # (ITERATIONS_TO_CONVERGE, Counter()))

if __name__ == '__main__':
    unittest.main()
