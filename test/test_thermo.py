import unittest
import sys, os, re
import forcebalance
import abc
import numpy
from __init__ import ForceBalanceTestCase
from collections import defaultdict, OrderedDict

class TestParser(ForceBalanceTestCase):
    def setUp(self):
        os.chdir(os.path.join(os.getcwd(), 'studies', '004_thermo'))
        input_file='test_parse.in'
        options, tgt_opts = forcebalance.parser.parse_inputs(input_file)
        forcefield = forcebalance.forcefield.FF(options)
        self.objective = forcebalance.objective.Objective(options, tgt_opts, forcefield)

    def test_lipid_parser(self):
        """Test for equality amongst multiple ways to parse lipid experimental data"""
        # Build a dictionary of target name : dataframes
        lipid_data = OrderedDict()
        for tgt in self.objective.Targets:
            if 'lipid' in tgt.name.lower():
                lipid_data[tgt.name] = tgt.Data
        # Double loop over different targets
        for i, ikey in enumerate(lipid_data.keys()):
            for j, jkey in enumerate(lipid_data.keys()):
                # Check column headings and row indices
                self.assertTrue(all(lipid_data[ikey].columns == lipid_data[jkey].columns), msg='\nColumn headings not equal for %s and %s' % (ikey, jkey))
                self.assertTrue(all(lipid_data[ikey].index == lipid_data[jkey].index), msg='\nRow indices not equal for %s and %s' % (ikey, jkey))
                # Make dictionary representation of dataframes
                dicti = lipid_data[ikey].to_dict()
                dictj = lipid_data[jkey].to_dict()
                # Here's where it gets complicated.
                # Loop over data columns.
                for column in dicti.keys():
                    dseti = defaultdict(set)
                    dsetj = defaultdict(set)
                    # For each data column, the dataframe contains a
                    # set of data which is keyed by the system index.
                    # Each row is further keyed by the subindex, but
                    # this test assumes that the subindices are
                    # irrelevant (equivalent to saying the ordering of
                    # rows - or the relative vertical position of data
                    # in cells across columns - is not important.  Not
                    # entirely true but anyway...)
                    for idx in dicti[column].keys():
                        dseti[idx[0]].add(dicti[column][idx])
                        dsetj[idx[0]].add(dictj[column][idx])
                    self.assertEqual(dseti, dsetj, msg='\n%s data column not equal for targets %s and %s' % (i, ikey, jkey))
        

if __name__ == '__main__':           
    unittest.main()
