from __future__ import absolute_import
from __init__ import ForceBalanceTestCase
import unittest
import forcebalance
import os, sys
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from forcebalance import optimizer as fbopt

class TestWaterTutorial(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.system('rm -rf test/files/test_liquid.run')
        os.system('cp -r test/files/test_liquid/ test/files/test_liquid.run/')
        os.chdir('test/files/test_liquid.run')

    def tearDown(self):
        os.chdir('..')
        os.system('rm -rf test_liquid.run')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check liquid target with existing simulation data"""
        if not sys.version_info <= (2,7):
            self.skipTest("Existing pickle file only works with Python 3")
        
        self.logger.debug("\nSetting input file to 'single.in'\n")
        input_file='single.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        forcefield  = FF(options)
        objective   = Objective(options, tgt_opts, forcefield)
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

        liquid_obj_value = optimizer.Objective.ObjDict['Liquid']['x']
        self.assertTrue(liquid_obj_value < 20, msg="\nLiquid objective function should give < 20 (about 17.23) total value.")

if __name__ == '__main__':
    unittest.main()
