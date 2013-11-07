import sys, os, shutil
import forcebalance.parser
import unittest
from __init__ import ForceBalanceTestCase

class TestParser(ForceBalanceTestCase):
    def test_parse_inputs_returns_tuple(self):
        """Check parse_inputs() returns type"""
        output = forcebalance.parser.parse_inputs('test/files/very_simple.in')
        self.assertEqual(type(output), tuple,
        msg = "\nExpected parse_inputs() to return a tuple, but got a %s instead" % type(output).__name__)
        self.assertEqual(type(output[0]), dict,
        msg = "\nExpected parse_inputs()[0] to be an options dictionary, got a %s instead" % type(output).__name__)
        self.assertEqual(type(output[1]), list,
        msg = "\nExpected parse_inputs()[1] to be a target list, got a %s instead" % type(output[1]).__name__)
        self.assertEqual(type(output[1][0]), dict,
        msg = "\nExpected parse_inputs()[1][0] to be a target dictionary, got a %s instead" % type(output[1]).__name__)

    def test_parse_inputs_generates_default_options(self):
        """Check parse_inputs() without arguments generates dictionary of default options"""
        output = forcebalance.parser.parse_inputs()

        defaults = forcebalance.parser.gen_opts_defaults
        defaults.update({'root':os.getcwd()})
        defaults.update({'input_file':None})
        target_defaults = forcebalance.parser.tgt_opts_defaults

        self.assertEqual(output[0], defaults,
        msg="\nparse_inputs() target options do not match those in forcebalance.parser.gen_opts_defaults")
        self.assertEqual(output[1][0], target_defaults,
        msg="\nparse_inputs() target options do not match those in forcebalance.parser.tgt_opts_defaults")

    def test_parse_inputs_yields_consistent_results(self):
        """Check parse_inputs() gives consistent results"""
        output1 = forcebalance.parser.parse_inputs('test/files/very_simple.in')
        output2 = forcebalance.parser.parse_inputs('test/files/very_simple.in')
        self.assertEqual(output1,output2)

        os.chdir('test/files')

        output3 = forcebalance.parser.parse_inputs('very_simple.in')
        output4 = forcebalance.parser.parse_inputs('very_simple.in')
        self.assertEqual(output3,output4)

        # directory change should lead to different result in output['root']
        self.assertNotEqual(output1,output3)

        # different parameters from the same file should yield different results
        shutil.copyfile('0.energy_force.in', 'test.in')
        output5 = forcebalance.parser.parse_inputs('test.in')
        shutil.copyfile('1.netforce_torque.in','test.in')
        output6 = forcebalance.parser.parse_inputs('test.in')
        self.assertNotEqual(output5,output6)
        os.remove('test.in')

if __name__ == '__main__':           
    unittest.main()
