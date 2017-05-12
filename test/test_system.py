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
from numpy import absolute

# expected results (mvals) taken from previous runs. Update this if it changes and seems reasonable (updated 10/24/13)
#EXPECTED_WATER_RESULTS = array([3.3192e-02, 4.3287e-02, 5.5072e-03, -4.5933e-02, 1.5499e-02, -3.7655e-01, 2.4720e-03, 1.1914e-02, 1.5066e-01])
EXPECTED_WATER_RESULTS = array([4.2370e-02, 3.1217e-02, 5.6925e-03, -4.8114e-02, 1.6735e-02, -4.1722e-01, 6.2716e-03, 4.6306e-03, 2.5960e-01])

# expected results (mvals) taken from previous runs. Update this if it changes and seems reasonable (updated 01/24/14)
EXPECTED_BROMINE_RESULTS = array([-0.305718, -0.12497])

# expected result (pvals) taken from ethanol GB parameter optimization. Update this if it changes and seems reasonable (updated 09/05/14)
EXPECTED_ETHANOL_RESULTS = array([1.2286e-01, 8.3624e-01, 1.0014e-01, 8.4533e-01, 1.8740e-01, 6.8820e-01, 1.4606e-01, 8.3518e-01])

# fail test if we take more than this many iterations to converge. Update this as necessary
ITERATIONS_TO_CONVERGE = 5

# expected results taken from previous runs. Update this if it changes and seems reasonable (updated 07/23/14)
EXPECTED_LIPID_RESULTS = array([-6.7553e-03, -2.4070e-02])

class TestWaterTutorial(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/001_water_tutorial')
        targets = tarfile.open('targets.tar.bz2','r')
        targets.extractall()
        targets.close()

    def tearDown(self):
        os.system('rm -rf results *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check water tutorial study runs without errors"""
        self.logger.debug("\nSetting input file to 'very_simple.in'\n")
        input_file='very_simple.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
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

        self.assertNdArrayEqual(EXPECTED_WATER_RESULTS,result,delta=0.001,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_WATER_RESULTS in test_system.py with these values")

        # Fail if calculation takes longer than previously to converge
        self.assertGreaterEqual(ITERATIONS_TO_CONVERGE, Counter(), msg="\nCalculation took longer than expected to converge (%d iterations vs previous of %d)" %\
        (ITERATIONS_TO_CONVERGE, Counter()))

class TestVoelzStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/009_voelz_nspe')

    def tearDown(self):
        os.system('rm -rf results *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check voelz study runs without errors"""
        self.logger.debug("\nSetting input file to 'options.in'\n")
        input_file='options.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        self.logger.debug("Creating forcefield using loaded options: ")
        forcefield  = FF(options)
        self.logger.debug(str(forcefield) + "\n")
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        self.logger.debug("Creating object using loaded options and forcefield: ")
        objective   = Objective(options, tgt_opts, forcefield)
        self.logger.debug(str(objective) + "\n")
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.logger.debug(str(optimizer) + "\n")
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()

        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

class TestBromineStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/003_liquid_bromine')

    def tearDown(self):
        os.system('rm -rf results *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check liquid bromine study converges to expected results"""
        self.logger.debug("\nSetting input file to 'options.in'\n")
        input_file='optimize.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        self.logger.debug("Creating forcefield using loaded options: ")
        forcefield  = FF(options)
        self.logger.debug(str(forcefield) + "\n")
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        self.logger.debug("Creating object using loaded options and forcefield: ")
        objective   = Objective(options, tgt_opts, forcefield)
        self.logger.debug(str(objective) + "\n")
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.logger.debug(str(optimizer) + "\n")
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()

        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

        self.assertNdArrayEqual(EXPECTED_BROMINE_RESULTS,result,delta=0.02,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_BROMINE_RESULTS in test_system.py with these values")

class TestThermoBromineStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/004_thermo_liquid_bromine')

    def tearDown(self):
        os.system('rm -rf results *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check liquid bromine study (Thermo target) converges to expected results"""
        self.logger.debug("\nSetting input file to 'optimize.in'\n")
        input_file='optimize.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        self.logger.debug("Creating forcefield using loaded options: ")
        forcefield  = FF(options)
        self.logger.debug(str(forcefield) + "\n")
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        self.logger.debug("Creating object using loaded options and forcefield: ")
        objective   = Objective(options, tgt_opts, forcefield)
        self.logger.debug(str(objective) + "\n")
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.logger.debug(str(optimizer) + "\n")
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()

        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

        self.assertNdArrayEqual(EXPECTED_BROMINE_RESULTS,result,delta=0.02,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_BROMINE_RESULTS in test_system.py with these values")

class TestLipidStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/010_lipid_study')

    def tearDown(self):
        os.system('rm -rf results *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check lipid tutorial study runs without errors"""
        self.logger.debug("\nSetting input file to 'options.in'\n")
        input_file='simple.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        forcefield = FF(options)
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        objective = Objective(options, tgt_opts, forcefield)
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

        self.assertNdArrayEqual(absolute(EXPECTED_LIPID_RESULTS),absolute(result),delta=0.010,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_LIPID_RESULTS in test_system.py with these values (%s)" % result)

        # Fail if calculation takes longer than previously to converge
        self.assertGreaterEqual(ITERATIONS_TO_CONVERGE, Counter(), msg="\nCalculation took longer than expected to converge (%d iterations vs previous of %d)" %\
        (ITERATIONS_TO_CONVERGE, Counter()))

class TestImplicitSolventHFEStudy(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        os.chdir('studies/012_implicit_solvent_hfe')

    def tearDown(self):
        os.system('rm -rf results *.bak *.tmp')
        super(ForceBalanceTestCase,self).tearDown()

    def runTest(self):
        """Check implicit hydration free energy study (Hydration target) converges to expected results"""
        self.logger.debug("\nSetting input file to 'optimize.in'\n")
        input_file='optimize.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        self.assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        self.assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            self.assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")

        ## The force field component of the project
        self.logger.debug("Creating forcefield using loaded options: ")
        forcefield  = FF(options)
        self.logger.debug(str(forcefield) + "\n")
        self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")

        ## The objective function
        self.logger.debug("Creating object using loaded options and forcefield: ")
        objective   = Objective(options, tgt_opts, forcefield)
        self.logger.debug(str(objective) + "\n")
        self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        self.logger.debug(str(optimizer) + "\n")
        self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()

        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result) + '\n')

        self.assertNdArrayEqual(EXPECTED_ETHANOL_RESULTS,forcefield.create_pvals(result),delta=0.02,
                                msg="\nCalculation results have changed from previously calculated values.\n"
                                "If this seems reasonable, update EXPECTED_ETHANOL_RESULTS in test_system.py with these values")

if __name__ == '__main__':
    unittest.main()
