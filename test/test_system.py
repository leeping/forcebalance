import unittest
import os, sys
import tarfile
from __init__ import ForceBalanceTestCase, TestValues
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from collections import OrderedDict

class TestTutorial(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        self.cwd = os.getcwd()
        os.chdir('studies/001_water_tutorial')
        if not os.path.isdir('targets'):
            targets = tarfile.open('targets.tar.bz2','r')
            targets.extractall()
            targets.close()

    def tearDown(self):
        super(ForceBalanceTestCase,self).tearDown()
        os.system('rm -rf results targets backups temp')
        os.chdir(self.cwd)

    def runTest(self):
        """Check whether tutorial runs and output has not changed from known baseline"""
        input_file='very_simple.in'

        ## The general options and target options that come from parsing the input file
        options, tgt_opts = parse_inputs(input_file)

        for key in TestValues.water_options.iterkeys():
            self.assertEqual(options[key],TestValues.water_options[key], msg="\nunexpected value for options['%s']" % key)

        ## The force field component of the project
        forcefield  = FF(options)
        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        ## The optimizer component of the project
        optimizer   = Optimizer(options, objective, forcefield)
        ## Actually run the optimizer.
        optimizer.Run()

if __name__ == '__main__':
    unittest.main()