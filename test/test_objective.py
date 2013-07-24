import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase

class TestImplemented(ForceBalanceTestCase):
    def test_implemented_targets_derived_from_target(self):
        """Check classes listed in Implemented_Targets are derived from Target"""
        for key in forcebalance.objective.Implemented_Targets.iterkeys():
            self.assertTrue(issubclass(forcebalance.objective.Implemented_Targets[key],forcebalance.target.Target))
    
    def test_no_unlisted_classes_derived_from_Target(self):
        """Check for unknown omissions from Implemented_Targets"""
        forcebalance_modules=[module[:-3] for module in os.listdir(forcebalance.__path__[0])
                     if re.compile(".*\.py$").match(module)
                     and module not in ["__init__.py"]]
        for module in forcebalance_modules:
            m = __import__('forcebalance.' + module)
            objects = dir(eval('m.' + module))
            for object in objects:
                object = eval('m.'+module+'.'+object)
                if type(object) == abc.ABCMeta:
                    implemented = [i for i in forcebalance.objective.Implemented_Targets.itervalues()]
                    # list of documented exceptions
                    exclude = ['Target',
                               'AbInitio',
                               'Interaction',
                               'Interaction_GMX',
                               'Liquid',
                               'BindingEnergy',
                               'LeastSquares',
                               'Vibration',
                               'Moments']
                    if object not in implemented and object.__name__ not in exclude:
                        self.fail("Unknown class '%s' not listed in Implemented_Targets" % object.__name__)

class TestPenalty(ForceBalanceTestCase):
    def setUp(self):
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        os.chdir(self.options['root'])

        try:
            self.ff = forcebalance.forcefield.FF(self.options)
            self.np=self.ff.np
        except:
            self.skipTest("Unable to create forcefield needed for penalty tests")

        self.penalties = []
        for ptype in forcebalance.objective.Penalty.Pen_Names.keys():
            penalty = forcebalance.objective.Penalty(ptype,
                                self.ff,
                                self.options['penalty_additive'],
                                self.options['penalty_multiplicative'],
                                self.options['penalty_hyperbolic_b'],
                                self.options['penalty_alpha'])
            self.penalties.append(penalty)

    def test_penalty_compute(self):
        """Check penalty computation functions"""
        objective = {'G': numpy.zeros((9)),
         'H': numpy.diag((1,)*9),
         'X': 1}
        for penalty in self.penalties:
            result=penalty.compute([1]*self.np, objective)
            self.assertEqual(tuple, type(result))
            # more tests go here

class TestObjective(ForceBalanceTestCase):
    def setUp(self):
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd() + '/test/files',
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        os.chdir(self.options['root'])

        self.tgt_opts = forcebalance.parser.tgt_opts_defaults.copy()

    def runTest(self):
        pass

if __name__ == '__main__':           
    unittest.main()
