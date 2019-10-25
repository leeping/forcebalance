from __future__ import absolute_import
from builtins import str
from builtins import object
import sys, os, re
import forcebalance
import abc
import numpy
import pytest

def test_implemented_targets_derived_from_target():
    """Check classes listed in Implemented_Targets are derived from Target"""
    for key in forcebalance.objective.Implemented_Targets.keys():
        print("Assert %s is subclass of target\n" % str(forcebalance.objective.Implemented_Targets[key]))
        assert issubclass(forcebalance.objective.Implemented_Targets[key],forcebalance.target.Target)

def test_no_unlisted_classes_derived_from_Target():
    """Check for unknown omissions from Implemented_Targets

    Check to make sure any classes derived from Target are either
    listed in Implemented_Targets or in the exclusion list in this
    test case
    """
    forcebalance_modules=[module[:-3] for module in os.listdir(forcebalance.__path__[0])
                 if re.compile(".*\.py$").match(module)
                 and module not in ["__init__.py"]]
    for module in forcebalance_modules:
        # LPW: I don't think dcdlib should be imported this way.
        print(module)
        if module == "_dcdlib": continue
        m = __import__('forcebalance.' + module)
        objs = dir(eval('m.' + module))
        print(objs)
        for obj in objs:
            obj = eval('m.'+module+'.'+obj)
            if type(obj) == abc.ABCMeta:
                implemented = [i for i in forcebalance.objective.Implemented_Targets.values()]
                # list of documented exceptions
                # Basically, platform-independent targets are excluded.
                exclude = ['Target',
                           'AbInitio',
                           'Interaction',
                           'Interaction_GMX',
                           'Liquid',
                           'Lipid',
                           'BindingEnergy',
                           'LeastSquares',
                           'Vibration',
                           'Thermo',
                           'Hydration',
                           'Moments']
                print(obj)
                if obj not in implemented and obj.__name__ not in exclude:
                    pytest.fail("Unknown class '%s' not listed in Implemented_Targets" % obj.__name__)

class TestPenalty:
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        os.chdir(cls.options['root'])

        cls.ff = forcebalance.forcefield.FF(cls.options)
        cls.np=cls.ff.np

        cls.penalties = []
        for ptype in forcebalance.objective.Penalty.Pen_Names.keys():
            penalty = forcebalance.objective.Penalty(ptype,
                                cls.ff,
                                cls.options['penalty_additive'],
                                cls.options['penalty_multiplicative'],
                                cls.options['penalty_hyperbolic_b'],
                                cls.options['penalty_alpha'])
            cls.penalties.append(penalty)

    def test_penalty_compute(self):
        """Check penalty computation functions"""
        objective = {'G': numpy.zeros((9)),
         'H': numpy.diag((1,)*9),
         'X': 1}
        for penalty in self.penalties:
            result=penalty.compute([1]*self.np, objective)
            assert isinstance(result,tuple)
            # more tests go here

class ObjectiveTests(object):
    def test_target_zero_order_terms(self):
        """Check zero order target terms"""
        obj = self.objective.Target_Terms(numpy.array([.5]*self.ff.np), Order=0)
        assert isinstance(obj, dict)
        assert "X" in obj
        assert "G" in obj
        assert "H" in obj
        assert int(obj["X"]) != 0
        assert obj["G"].any() == False
        assert (obj["H"] == numpy.diag([1]*self.ff.np)).all()

    def test_target_first_order_terms(self):
        """Check first order target terms"""
        obj = self.objective.Target_Terms(numpy.array([.5]*self.ff.np), Order=1)
        assert isinstance(obj, dict)
        assert "X" in obj
        assert "G" in obj
        assert "H" in obj

    def test_target_second_order_terms(self):
        """Check second order target terms"""
        obj = self.objective.Target_Terms(numpy.array([.5]*self.ff.np), Order=2)
        assert isinstance(obj, dict)
        assert "X" in obj
        assert "G" in obj
        assert "H" in obj

    def test_indicate(self):
        """Check objective.indicate() runs without errors"""
        self.objective.Indicate()

class TestWaterObjective(ObjectiveTests):
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})
        os.chdir(cls.options['root'])

        print("\nUsing the following options:\n%s\n" % str(cls.options))

        cls.tgt_opts = [ forcebalance.parser.tgt_opts_defaults.copy() ]
        cls.tgt_opts[0].update({"type" : "ABINITIO_GMX", "name" : "cluster-06"})
        cls.ff = forcebalance.forcefield.FF(cls.options)

        cls.objective = forcebalance.objective.Objective(cls.options, cls.tgt_opts,cls.ff)

    def shortDescription(self):
        return super(TestWaterObjective, self).shortDescription() + " (AbInitio_GMX target)"

class TestBromineObjective(ObjectiveTests):
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['bro.itp']})
        os.chdir(cls.options['root'])

        print("\nUsing the following options:\n%s\n" % str(cls.options))

        cls.tgt_opts = [ forcebalance.parser.tgt_opts_defaults.copy() ]
        cls.tgt_opts[0].update({"type" : "LIQUID_GMX", "name" : "LiquidBromine"})
        cls.ff = forcebalance.forcefield.FF(cls.options)

        cls.objective = forcebalance.objective.Objective(cls.options, cls.tgt_opts,cls.ff)

    def shortDescription(self):
        return super(TestBromineObjective, self).shortDescription() + " (Liquid_GMX target)"
