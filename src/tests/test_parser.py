import os
import shutil
import forcebalance.parser
from .__init__ import ForceBalanceTestCase

class TestParser(ForceBalanceTestCase):

    def test_parse_inputs_returns_tuple(self):
        """Check parse_inputs() returns type"""
        output = forcebalance.parser.parse_inputs('files/very_simple.in')
        assert isinstance(output, tuple), "\nExpected parse_inputs() to return a tuple, but got a %s instead" % type(output).__name__
        assert isinstance(output[0], dict), "\nExpected parse_inputs()[0] to be an options dictionary, got a %s instead" % type(output).__name__
        assert isinstance(output[1], list), "\nExpected parse_inputs()[1] to be a target list, got a %s instead" % type(output[1]).__name__
        assert isinstance(output[1][0], dict), "\nExpected parse_inputs()[1][0] to be a target dictionary, got a %s instead" % type(output[1]).__name__

    def test_parse_inputs_generates_default_options(self):
        """Check parse_inputs() without arguments generates dictionary of default options"""
        output = forcebalance.parser.parse_inputs()
        defaults = forcebalance.parser.gen_opts_defaults
        defaults.update({'root':os.getcwd()})
        defaults.update({'input_file':None})
        target_defaults = forcebalance.parser.tgt_opts_defaults
        assert output[0] == defaults, "\nparse_inputs() target options do not match those in forcebalance.parser.gen_opts_defaults"
        assert output[1][0] == target_defaults, "\nparse_inputs() target options do not match those in forcebalance.parser.tgt_opts_defaults"

    def test_parse_inputs_yields_consistent_results(self):
        """Check parse_inputs() gives consistent results"""
        output1 = forcebalance.parser.parse_inputs('files/very_simple.in')
        output2 = forcebalance.parser.parse_inputs('files/very_simple.in')
        assert output1 == output2
        os.chdir('files')
        output3 = forcebalance.parser.parse_inputs('very_simple.in')
        output4 = forcebalance.parser.parse_inputs('very_simple.in')
        assert output3 == output4
        # directory change should lead to different result in output['root']
        assert output1 != output3
        # different parameters from the same file should yield different results
        shutil.copyfile('0.energy_force.in', 'test.in')
        output5 = forcebalance.parser.parse_inputs('test.in')
        shutil.copyfile('1.netforce_torque.in','test.in')
        output6 = forcebalance.parser.parse_inputs('test.in')
        assert output5 != output6
        os.remove('test.in')
