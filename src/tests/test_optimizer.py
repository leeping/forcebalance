from __future__ import absolute_import
# from __init__ import ForceBalanceTestCase
# import unittest
import numpy
import forcebalance
import os, sys
import tarfile
import logging

logger = logging.getLogger("test")

# class TestOptimizer(ForceBalanceTestCase):
class TestOptimizer:
    # def setUp(self):
        # super(ForceBalanceTestCase,self).setUp()
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, '../../studies/001_water_tutorial'))
        # input_file='very_simple.in'
        # targets = tarfile.open('targets.tar.bz2','r')
        # targets.extractall()
        # targets.close()
        #
        # options, tgt_opts = forcebalance.parser.parse_inputs(input_file)
        #
        # options.update({'writechk':'checkfile.tmp'})
        #
        # forcefield  = forcebalance.forcefield.FF(options)
        # objective   = forcebalance.objective.Objective(options, tgt_opts, forcefield)
        # try: optimizer   = forcebalance.optimizer.Optimizer(options, objective, forcefield)
        # except: fail("\nCouldn't create optimizer")

    # def tearDown(self):
    @classmethod
    def teardown_class(cls):
        os.system('rm -rf result *.bak *.tmp')
        # super(ForceBalanceTestCase,self).tearDown()

    # def runTest(self):
    def test_optimizer(self):
        # moved from setUp
        input_file='very_simple.in'
        targets = tarfile.open('targets.tar.bz2','r')
        targets.extractall()
        targets.close()

        options, tgt_opts = forcebalance.parser.parse_inputs(input_file)

        options.update({'writechk':'checkfile.tmp'})

        forcefield  = forcebalance.forcefield.FF(options)
        objective   = forcebalance.objective.Objective(options, tgt_opts, forcefield)
        try: optimizer   = forcebalance.optimizer.Optimizer(options, objective, forcefield)
        except: fail("\nCouldn't create optimizer")

        optimizer.writechk()
        # self.assertTrue(os.path.isfile(options['writechk']),
        # msg="\nOptimizer.writechk() didn't create expected file at %s" % options['writechk'])
        assert os.path.isfile(options['writechk']), "Optimizer.writechk() didn't create expected file at %s " % options['writechk']
        read = optimizer.readchk()
        # self.assertEqual(type(read), dict)
        assert isinstance(read, dict)

# if __name__ == '__main__':
#     unittest.main()
