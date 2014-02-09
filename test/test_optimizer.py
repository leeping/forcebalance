from __init__ import ForceBalanceTestCase
import unittest
import numpy
import forcebalance
import os
import tarfile
import logging

logger = logging.getLogger("test")

class TestOptimizer(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/001_water_tutorial')
        self.input_file='very_simple.in'
        targets = tarfile.open('targets.tar.bz2','r')
        targets.extractall()
        targets.close()

        self.options, self.tgt_opts = forcebalance.parser.parse_inputs(self.input_file)

        self.options.update({'writechk':'checkfile.tmp'})

        self.forcefield  = forcebalance.forcefield.FF(self.options)
        self.objective   = forcebalance.objective.Objective(self.options, self.tgt_opts, self.forcefield)
        try: self.optimizer   = forcebalance.optimizer.Optimizer(self.options, self.objective, self.forcefield)
        except: self.fail("\nCouldn't create optimizer")

    def tearDown(self):
        os.system('rm -rf result *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        self.optimizer.writechk()
        self.assertTrue(os.path.isfile(self.options['writechk']),
        msg="\nOptimizer.writechk() didn't create expected file at %s" % self.options['writechk'])
        read = self.optimizer.readchk()
        self.assertEqual(type(read), dict)

if __name__ == '__main__':           
    unittest.main()
