import unittest
import os, sys
import tarfile
from __init__ import ForceBalanceTestCase, TestValues
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from collections import OrderedDict

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
        """Check tutorial runs without errors"""
        input_file='very_simple.in'

        ## The general options and target options that come from parsing the input file
        options, tgt_opts = parse_inputs(input_file)

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
