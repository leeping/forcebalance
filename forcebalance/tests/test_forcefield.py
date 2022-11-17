from __future__ import absolute_import
from builtins import str
from builtins import object
import os
import shutil
import forcebalance
import forcebalance.forcefield as forcefield
from .__init__ import ForceBalanceTestCase
import numpy as np

class FFTests(object):
    """Tests common to all forcefields. Note that to prevent this class from being run on its own
    by the Test Runner, we do not subclass ForceBalanceTestCase. The actual forcefield instance
    being tested needs to be provided by subclasses"""

    def test_FF_yields_consistent_results(self):
        """Check whether multiple calls to FF yield the same result"""
        self.logger.debug("\nChecking consistency of ForceField constructor\n")
        assert forcefield.FF(self.options) == forcefield.FF(self.options), \
            "Got two different forcefields despite using the same options as input"

    def test_make_function_return_value(self):
        """Check that make() return value meets expectation"""
        pvals = self.ff.pvals0

        self.logger.debug("Running forcefield.make() with zero vector should not change pvals... ")
        new_pvals = np.array(self.ff.make(np.zeros(self.ff.np)))
        assert pvals.size == new_pvals.size
        # assert (pvals == new_pvals).all(), "make() should produce unchanged pvals when given zero vector"
        msg="make() should produce unchanged pvals when given zero vector"
        np.testing.assert_array_almost_equal(pvals, new_pvals, decimal=5, err_msg=msg)
        self.logger.debug("ok\n")

        self.logger.debug("make() should return different values when passed in nonzero pval matrix... ")
        new_pvals = np.array(self.ff.make(np.ones(self.ff.np)))
        assert pvals.size == new_pvals.size
        # given arbitrary nonzero input, make should return new pvals
        assert not (pvals==new_pvals).all(), "make() returned unchanged pvals even when given nonzero matrix"
        self.logger.debug("ok\n")

        self.logger.debug("make(use_pvals=True) should return the same pvals... ")
        new_pvals = np.array(self.ff.make(np.ones(self.ff.np), use_pvals=True))
        # assert (np.ones(self.ff.np) == new_pvals).all(), "make() did not return input pvals with use_pvals=True"
        msg = "make() did not return input pvals with use_pvals=True"
        np.testing.assert_array_almost_equal(np.ones(self.ff.np), new_pvals, decimal=5, err_msg=msg)
        self.logger.debug("ok\n")

        os.remove(os.path.join(self.options['root'], self.ff.fnms[0]))

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


class TestWaterFF(ForceBalanceTestCase, FFTests):
    """Test FF class using water options and forcefield (text forcefield input)
    This test case also acts as a base class for other forcefield test cases.
    Override the setUp() to run tests on a different forcefield"""
    def setup_method(self, method):
        super(TestWaterFF, self).setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, 'files'))
        # options used in 001_water_tutorial
        self.logger.debug("\nSetting up options...\n")
        self.options = forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['water.itp']})
        self.logger.debug(str(self.options) + '\n')
        self.logger.debug("Creating forcefield using above options... ")
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.logger.debug("ok\n")

class TestXmlFF(ForceBalanceTestCase, FFTests):
    """Test FF class using dms.xml forcefield input"""
    def setup_method(self, method):
        super(TestXmlFF, self).setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, 'files'))
        # options from 2013 tutorial
        self.logger.debug("Setting up options...\n")
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['dms.xml']})
        self.logger.debug(str(self.options) + '\n')
        self.logger.debug("Creating forcefield using above options... ")
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.logger.debug("ok\n")

class TestXmlScriptFF(ForceBalanceTestCase):
    """Test FF class with XmlScript using TIP3G2w.xml forcefield input"""
    def setup_method(self, method):
        super(TestXmlScriptFF, self).setup_method(method)
        self.logger.debug("Setting up options...\n")
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, 'files'))
        # Load the base force field file
        self.ff = forcefield.FF.fromfile('forcefield/TIP3G2w.xml')
        # Load mathematical parameter values corresponding to a known output
        self.mvals = np.loadtxt('XmlScript_out/mvals.txt')
        # Load the known output force field
        self.ff_ref = forcefield.FF.fromfile('XmlScript_out/TIP3G2w_out_ref.xml')

    # def tearDown(self):
    def teardown_method(self):
        tmpfolder = os.path.join(self.cwd, 'files', 'TIP3G2w.xml')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        # os.system('rm -rf TIP3G2w.xml')
        super(TestXmlScriptFF, self).teardown_method()

    def test_make_function_output(self):
        """Check make() function creates expected force field file containing XML Script"""
        os.chdir(os.path.join(self.cwd, 'files', 'XmlScript_out'))
        # Create the force field with mathematical parameter values and
        # make sure it matches the known reference
        self.ff.make(self.mvals)
        ff_out = forcefield.FF.fromfile('TIP3G2w.xml')
        assert self.ff_ref == ff_out, "make() produced a different output force field"

class TestGbsFF(ForceBalanceTestCase, FFTests):
    """Test FF class using gbs forcefield input"""
    def setup_method(self, method):
        super(TestGbsFF, self).setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(self.cwd, 'files'))
        self.logger.debug("Setting up options...\n")
        self.options=forcebalance.parser.gen_opts_defaults.copy()
        self.options.update({
                'root': os.getcwd(),
                'penalty_additive': 0.01,
                'jobtype': 'NEWTON',
                'forcefield': ['cc-pvdz-overlap-original.gbs']})
        self.logger.debug(str(self.options) + '\n')
        self.logger.debug("Creating forcefield using above options... ")
        self.ff = forcefield.FF(self.options)
        self.ffname = self.options['forcefield'][0][:-3]
        self.filetype = self.options['forcefield'][0][-3:]
        self.logger.debug("ok\n")

    def test_find_spacings(self):
        """Check find_spacings function"""
        self.logger.debug("Running forcefield.find_spacings()...\n")
        spacings = self.ff.find_spacings()

        assert (self.ff.np)*(self.ff.np-1)/2 >= len(spacings.keys())
        assert isinstance(spacings, dict)
