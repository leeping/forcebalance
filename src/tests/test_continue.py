from __future__ import absolute_import
from builtins import str
# import unittest
import os, sys, shutil
import tarfile
# from .__init__ import ForceBalanceTestCase
from forcebalance.nifty import printcool_dictionary
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer, Counter
from collections import OrderedDict
from numpy import array

# class TestWaterTutorial(ForceBalanceTestCase):
class TestWaterTutorial:
    # def setUp(self):
    @classmethod
    def setup_class(cls):
        # super(ForceBalanceTestCase,self).setUp()

        # os.system('rm -rf tests/files/test_continue.run')
        # os.makedirs('tests/files/test_continue.run')
        # os.system('cp -r ../../studies/001_water_tutorial/forcefield tests/files/test_continue.run/forcefield')
        # os.system('cp -r ../../studies/001_water_tutorial/targets tests/files/test_continue.run/targets')
        # os.chdir('test/files')
        # os.system('cp -r test_continue.sav test_continue.in test_continue.tmp test_continue.run')
        # os.chdir('test_continue.run')

        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        # copy folder 'files/test_liquid' into a new folder 'files/test_liquid.run'
        os.chdir(os.path.join(cls.cwd, 'files'))
        tmpfolder = os.path.join(cls.cwd, 'files', 'test_continue.run')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        os.mkdir(tmpfolder)
        source_folder = os.path.join(cls.cwd, '../../studies/001_water_tutorial')
        shutil.copytree(os.path.join(source_folder, 'forcefield'), os.path.join(tmpfolder, 'forcefield'))
        shutil.copytree(os.path.join(source_folder, 'targets'), os.path.join(tmpfolder, 'targets'))
        # os.system('cp -r test_continue.sav test_continue.in test_continue.tmp test_continue.run')
        shutil.copy("test_continue.sav", tmpfolder)
        shutil.copy("test_continue.in", tmpfolder)
        shutil.copytree("test_continue.tmp", os.path.join(tmpfolder,"test_continue.tmp"))
        os.chdir(tmpfolder)

    # def tearDown(self):
    @classmethod
    def teardown_class(cls):
        # os.chdir('..')
        # os.system('rm -rf test_continue.run')
        # super(ForceBalanceTestCase,self).tearDown()
        # remove temporary folder 'files/test_liquid.run'
        tmpfolder = os.path.join(cls.cwd, 'files', 'test_continue.run')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)

    # def runTest(self):
    def test_continue(self):
        """Check continuation from a previous run"""
        # if not sys.version_info <= (2,7):
        #     skipTest("Existing pickle file only works with Python 3")
        print("Setting input file to 'test_continue.in'")
        input_file='test_continue.in'

        ## The general options and target options that come from parsing the input file
        print("Parsing inputs...")
        options, tgt_opts = parse_inputs(input_file)
        options['continue'] = True
        print("options:%s\ntgt_opts:%s\n" % (str(options), str(tgt_opts)))

        assert isinstance(options, dict), "Parser gave incorrect type for options"
        assert isinstance(tgt_opts, list), "Parser gave incorrect type for tgt_opts"
        # assertEqual(dict,type(options), msg="\nParser gave incorrect type for options")
        # assertEqual(list,type(tgt_opts), msg="\nParser gave incorrect type for tgt_opts")
        for target in tgt_opts:
            # assertEqual(dict, type(target), msg="\nParser gave incorrect type for target dict")
            assert isinstance(target, dict), "Parser gave incorrect type for target dict"

        ## The force field component of the project
        forcefield  = FF(options)
        # self.assertEqual(FF, type(forcefield), msg="\nExpected forcebalance forcefield object")
        assert isinstance(forcefield, FF), "Expected forcebalance forcefield object"

        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        # self.assertEqual(Objective, type(objective), msg="\nExpected forcebalance objective object")
        assert isinstance(objective, Objective), "Expected forcebalance objective object"

        ## The optimizer component of the project
        print("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        # self.assertEqual(Optimizer, type(optimizer), msg="\nExpected forcebalance optimizer object")
        assert isinstance(optimizer, Optimizer), "Expected forcebalance optimizer object"
        print(str(optimizer))

        ## Actually run the optimizer.
        print("Done setting up! Running optimizer...")
        result = optimizer.Run()
        print("Optimizer finished. Final results:")
        print(str(result))

        # self.assertEqual(optimizer.iterinit, 2, msg="\nInitial iteration counter is incorrect")
        # self.assertEqual(optimizer.iteration, 2, msg="\nFinal iteration counter is incorrect")
        assert optimizer.iterinit == 2, "Initial iteration counter is incorrect"
        assert optimizer.iteration == 2, "Final iteration counter is incorrect"

        # self.assertNdArrayEqual(EXPECTED_WATER_RESULTS,result,delta=0.001,
        #                         msg="\nCalculation results have changed from previously calculated values.\n"
        #                         "If this seems reasonable, update EXPECTED_WATER_RESULTS in test_system.py with these values")
        # # Fail if calculation takes longer than previously to converge
        # self.assertGreaterEqual(ITERATIONS_TO_CONVERGE, Counter(), msg="\nCalculation took longer than expected to converge (%d iterations vs previous of %d)" %\
        # (ITERATIONS_TO_CONVERGE, Counter()))

# if __name__ == '__main__':
#     unittest.main()
