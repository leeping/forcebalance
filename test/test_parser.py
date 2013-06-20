import sys, os
import forcebalance.parser
import unittest
from __init__ import ForceBalanceTestCase

class TestParser(ForceBalanceTestCase):
    def test_parse_inputs_returns_tuple(self):
        """Check parse_inputs() returns a tuple"""
        output = forcebalance.parser.parse_inputs('studies/001_water_tutorial/very_simple.in')
        self.assertEqual(type(output), tuple)

    def test_parse_inputs_generates_default_options(self):
        """Check parse_inputs() without arguments generates default options"""
        output = forcebalance.parser.parse_inputs()[0]
        defaults = forcebalance.parser.gen_opts_defaults
        defaults.update({'root':os.getcwd()})
        self.assertEqual(output, defaults)

    def test_parse_inputs_yields_consistent_results(self):
        """Check parse_inputs() always returns the same output given the same input"""
        output1 = forcebalance.parser.parse_inputs('studies/001_water_tutorial/very_simple.in')
        output2 = forcebalance.parser.parse_inputs('studies/001_water_tutorial/very_simple.in')
        self.assertEqual(output1,output2)

        cwd=os.getcwd()
        os.chdir('studies/001_water_tutorial')

        output3 = forcebalance.parser.parse_inputs('very_simple.in')
        output4 = forcebalance.parser.parse_inputs('very_simple.in')
        self.assertEqual(output3,output4)

        # directory change should lead to different result in output['root']
        self.assertNotEqual(output1,output3)
        os.chdir(cwd)

if __name__ == '__main__':           
    unittest.main()
