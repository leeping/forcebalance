from __future__ import absolute_import
from builtins import str
import os, sys, shutil
from .__init__ import ForceBalanceTestCase
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer, Counter
import pytest

class TestWaterTutorial(ForceBalanceTestCase):
    def setup_method(self, method):
        super(TestWaterTutorial, self).setup_method(method)
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        # copy folder 'files/test_liquid' into a new folder 'files/test_liquid.run'
        files_folder = os.path.join(self.cwd, 'files')
        tmpfolder = os.path.join(self.cwd, 'files', 'test_continue.run')
        source_folder = os.path.join(self.cwd, '../../studies/001_water_tutorial')

        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        os.mkdir(tmpfolder)
        
        shutil.copytree(os.path.join(source_folder, 'forcefield'), os.path.join(tmpfolder, 'forcefield'))
        shutil.copytree(os.path.join(source_folder, 'targets'), os.path.join(tmpfolder, 'targets'))
        shutil.copy(os.path.join(files_folder, "test_continue.sav"), tmpfolder)
        shutil.copy(os.path.join(files_folder, "test_continue.in"), tmpfolder)
        shutil.copytree(os.path.join(files_folder, "test_continue.tmp"), os.path.join(tmpfolder,"test_continue.tmp"))
        os.chdir(tmpfolder)

    def teardown_method(self):
        tmpfolder = os.path.join(self.cwd, 'files', 'test_continue.run')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        super(TestWaterTutorial, self).teardown_method()

    def test_continue(self):
        """Check continuation from a previous run"""
        if sys.version_info < (3,0):
            pytest.skip("Existing pickle file only works with Python 3")
        self.logger.debug("\nSetting input file to 'test_continue.in'\n")
        input_file='test_continue.in'

        ## The general options and target options that come from parsing the input file
        self.logger.debug("Parsing inputs...\n")
        options, tgt_opts = parse_inputs(input_file)
        options['continue'] = True
        self.logger.debug("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        assert isinstance(options, dict), "Parser gave incorrect type for options"
        assert isinstance(tgt_opts, list), "Parser gave incorrect type for tgt_opts"
        for target in tgt_opts:
            assert isinstance(target, dict), "Parser gave incorrect type for target dict"

        ## The force field component of the project
        forcefield  = FF(options)
        assert isinstance(forcefield, FF), "Expected forcebalance forcefield object"

        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        assert isinstance(objective, Objective), "Expected forcebalance objective object"

        ## The optimizer component of the project
        self.logger.debug("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        assert isinstance(optimizer, Optimizer), "Expected forcebalance optimizer object"
        self.logger.debug(str(optimizer)+'\n')

        ## Actually run the optimizer.
        self.logger.debug("Done setting up! Running optimizer...\n")
        result = optimizer.Run()
        self.logger.debug("\nOptimizer finished. Final results:\n")
        self.logger.debug(str(result)+'\n')

        assert optimizer.iterinit == 2, "Initial iteration counter is incorrect"
        assert optimizer.iteration == 2, "Final iteration counter is incorrect"

        # self.assertNdArrayEqual(EXPECTED_WATER_RESULTS,result,delta=0.001,
        #                         msg="\nCalculation results have changed from previously calculated values.\n"
        #                         "If this seems reasonable, update EXPECTED_WATER_RESULTS in test_system.py with these values")
        # # Fail if calculation takes longer than previously to converge
        # self.assertGreaterEqual(ITERATIONS_TO_CONVERGE, Counter(), msg="\nCalculation took longer than expected to converge (%d iterations vs previous of %d)" %\
        # (ITERATIONS_TO_CONVERGE, Counter()))

