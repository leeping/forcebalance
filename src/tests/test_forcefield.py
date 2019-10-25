from __future__ import absolute_import
from builtins import str
from builtins import object
import sys, os
import forcebalance
import forcebalance.forcefield as forcefield
import numpy as np
from copy import deepcopy

class FFTests(object):
    """Tests common to all forcefields. Note that to prevent this class from being run on its own
    by the Test Runner, we do not subclass ForceBalanceTestCase. The actual forcefield instance
    being tested needs to be provided by subclasses"""

    def test_FF_yields_consistent_results(self):
        """Check whether multiple calls to FF yield the same result"""
        print("\nChecking consistency of ForceField constructor\n")
        assert forcefield.FF(self.options) == forcefield.FF(self.options), "Got two different forcefields despite using the same options as input"

    def test_make_function_return_value(self):
        """Check that make() return value meets expectation"""
        pvals = self.ff.pvals0

        print("Running forcefield.make() with zero vector should not change pvals... ")
        new_pvals = np.array(self.ff.make(np.zeros(self.ff.np)))
        assert pvals.size == new_pvals.size
        assert (pvals == new_pvals).all(), "make() should produce unchanged pvals when given zero vector"
        print("ok\n")

        print("make() should return different values when passed in nonzero pval matrix... ")
        new_pvals = np.array(self.ff.make(np.ones(self.ff.np)))
        assert pvals.size == new_pvals.size
        # given arbitrary nonzero input, make should return new pvals
        assert not (pvals==new_pvals).all(), "make() returned unchanged pvals even when given nonzero matrix"
        print("ok\n")

        print("make(use_pvals=True) should return the same pvals... ")
        new_pvals = np.array(self.ff.make(np.ones(self.ff.np),use_pvals=True))
        assert (np.ones(self.ff.np) == new_pvals).all(), "make() did not return input pvals with use_pvals=True"
        print("ok\n")

        os.remove(self.options['root'] + '/' + self.ff.fnms[0])

    def test_make_function_output(self):
        """Check make() function creates expected forcefield file"""

        # read a forcefield from the output of make([--0--])
        self.ff.make(np.zeros(self.ff.np))
        os.rename(self.ff.fnms[0], self.options['ffdir']+'/test_zeros.' + self.filetype)
        self.options['forcefield']=['test_zeros.'+ self.filetype]
        ff_zeros = forcefield.FF(self.options)
        assert self.ff == ff_zeros, "make([0]) produced a different output forcefield"
        os.remove(self.options['ffdir']+'/test_zeros.' + self.filetype)

        # read a forcefield from the output of make([--1--])
        self.ff.make(np.ones(self.ff.np))
        os.rename(self.ff.fnms[0], self.options['ffdir']+'/test_ones.' + self.filetype)
        self.options['forcefield']=['test_ones.'+ self.filetype]
        ff_ones = forcefield.FF(self.options)

        assert self.ff != ff_ones, "make([1]) produced an unchanged output forcefield"
        os.remove(self.options['ffdir']+'/test_ones.' + self.filetype)


class TestWaterFF(FFTests):
    """Test FF class using water options and forcefield (text forcefield input)
    This test case also acts as a base class for other forcefield test cases.
    Override the setUp() to run tests on a different forcefield"""
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        # options used in 001_water_tutorial
        print("\nSetting up options...\n")
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})
        print(str(cls.options) + '\n')

        print("Creating forcefield using above options... ")
        cls.ff = forcefield.FF(cls.options)
        cls.ffname = cls.options['forcefield'][0][:-3]
        cls.filetype = cls.options['forcefield'][0][-3:]
        print("ok\n")

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestWaterFF,self).shortDescription() + " (itp)"

class TestXmlFF(FFTests):
    """Test FF class using dms.xml forcefield input"""
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        # options from 2013 tutorial
        print("Setting up options...\n")
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})
        print(str(cls.options) + '\n')

        print("Creating forcefield using above options... ")
        cls.ff = forcefield.FF(cls.options)
        cls.ffname = cls.options['forcefield'][0][:-3]
        cls.filetype = cls.options['forcefield'][0][-3:]
        print("ok\n")

    def shortDescription(self):
        """Add XML to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestXmlFF,self).shortDescription() + " (xml)"

class TestXmlScriptFF:
    """Test FF class with XmlScript using TIP3G2w.xml forcefield input"""
    @classmethod
    def setup_class(cls):
        print("Setting up options...\n")
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        # Load the base force field file
        cls.ff = forcefield.FF.fromfile('forcefield/TIP3G2w.xml')
        # Load mathematical parameter values corresponding to a known output
        cls.mvals = np.loadtxt('XmlScript_out/mvals.txt')
        # Load the known output force field
        cls.ff_ref = forcefield.FF.fromfile('XmlScript_out/TIP3G2w_out_ref.xml')

    # def tearDown(self):
    def teardown_class(cls):
        os.system('rm -rf TIP3G2w.xml')

    def test_make_function_output(self):
        """Check make() function creates expected force field file containing XML Script"""
        os.chdir('XmlScript_out')
        # Create the force field with mathematical parameter values and
        # make sure it matches the known reference
        self.ff.make(self.mvals)
        ff_out = forcefield.FF.fromfile('TIP3G2w.xml')
        assert self.ff_ref == ff_out, "make() produced a different output force field"

class TestGbsFF(FFTests):
    """Test FF class using gbs forcefield input"""
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, 'files'))
        print("Setting up options...\n")
        cls.options=forcebalance.parser.gen_opts_defaults.copy()
        cls.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        print(str(cls.options) + '\n')

        print("Creating forcefield using above options... ")
        cls.ff = forcefield.FF(cls.options)
        cls.ffname = cls.options['forcefield'][0][:-3]
        cls.filetype = cls.options['forcefield'][0][-3:]
        print("ok\n")

    def test_find_spacings(self):
        """Check find_spacings function"""
        print("Running forcefield.find_spacings()...\n")
        spacings = self.ff.find_spacings()

        assert (self.ff.np)*(self.ff.np-1)/2 >= len(spacings.keys())
        assert isinstance(spacings, dict)

    def shortDescription(self):
        """Add gbs to test descriptions
        @override __init__.ForceBalanceTestCase.shortDescription()"""
        return super(TestGbsFF,self).shortDescription() + " (gbs)"
