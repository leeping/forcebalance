from __future__ import absolute_import
import os
import sys
import shutil
import forcebalance
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from forcebalance import optimizer as fbopt

class TestWaterTutorial:
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        # copy folder 'files/test_liquid' into a new folder 'files/test_liquid.run'
        os.chdir(os.path.join(cls.cwd, 'files'))
        tmpfolder = os.path.join(cls.cwd, 'files', 'test_liquid.run')
        source_folder = os.path.join(cls.cwd, 'files', 'test_liquid')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)
        shutil.copytree(source_folder, tmpfolder)
        os.chdir(tmpfolder)

    @classmethod
    def tearDown(cls):
        # remove temporary folder 'files/test_liquid.run'
        tmpfolder = os.path.join(cls.cwd, 'files', 'test_liquid.run')
        if os.path.isdir(tmpfolder):
            shutil.rmtree(tmpfolder)

    def test_liquid(self):
        """Check liquid target with existing simulation data"""
        # if not sys.version_info <= (2,7):
        #     self.skipTest("Existing pickle file only works with Python 3")

        print("Setting input file to 'single.in'")
        input_file='single.in'

        ## The general options and target options that come from parsing the input file
        print("Parsing inputs...")
        options, tgt_opts = parse_inputs(input_file)
        print("options:\n%s\n\ntgt_opts:\n%s\n\n" % (str(options), str(tgt_opts)))

        forcefield  = FF(options)
        objective   = Objective(options, tgt_opts, forcefield)
        ## The optimizer component of the project
        print("Creating optimizer: ")
        optimizer   = Optimizer(options, objective, forcefield)
        assert isinstance(optimizer, Optimizer), "Expected forcebalance optimizer object"
        print(str(optimizer))

        ## Actually run the optimizer.
        print("Done setting up! Running optimizer...")
        result = optimizer.Run()
        print("\nOptimizer finished. Final results:")
        print(str(result))

        liquid_obj_value = optimizer.Objective.ObjDict['Liquid']['x']
        assert liquid_obj_value < 20, "Liquid objective function should give < 20 (about 17.23) total value."

