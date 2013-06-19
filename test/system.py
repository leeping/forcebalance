import unittest
import os, sys
import tarfile
from __init__ import ForceBalanceTestCase
from forcebalance.parser import parse_inputs
from forcebalance.forcefield import FF
from forcebalance.objective import Objective
from forcebalance.optimizer import Optimizer
from collections import OrderedDict

class TestTutorial(ForceBalanceTestCase):
    def setUp(self):
        super(ForceBalanceTestCase,self).setUp()
        self.cwd = os.getcwd()
        os.chdir('studies/001_water_tutorial')
        if not os.path.isdir('targets'):
            targets = tarfile.open('targets.tar.bz2','r')
            targets.extractall()
            targets.close()

    def tearDown(self):
        super(ForceBalanceTestCase,self).tearDown()
        os.system('rm -rf results targets backups temp')
        os.chdir(self.cwd)

    def runTest(self):
        """Checks whether tutorial runs and output has not changed from known baseline"""
        input_file='very_simple.in'

        ## The general options and target options that come from parsing the input file
        options, tgt_opts = parse_inputs(input_file)

        baseline_options={'penalty_type': 'L2', 'print_gradient': 1, 'eig_lowerbound': 0.0001, 'error_tolerance': 0.0, 'scanindex_name': [], 'read_mvals': None, 'maxstep': 100, 'print_parameters': 1, 'penalty_hyperbolic_b': 1e-06, 'gmxsuffix': '', 'readchk': None, 'mintrust': 0.0, 'penalty_multiplicative': 0.0, 'convergence_step': 0.0001, 'adaptive_damping': 0.5, 'finite_difference_h': 0.001, 'wq_port': 0, 'verbose_options': 0, 'scan_vals': None, 'logarithmic_map': 0, 'writechk_step': 1, 'forcefield': ['water.itp'], 'use_pvals': 0, 'scanindex_num': [], 'normalize_weights': 1, 'adaptive_factor': 0.25, 'trust0': 0.1, 'penalty_additive': 0.01, 'gmxpath': '/usr/bin', 'writechk': None, 'print_hessian': 0, 'have_vsite': 0, 'tinkerpath': '', 'ffdir': 'forcefield', 'constrain_charge': 0, 'convergence_gradient': 0.0001, 'convergence_objective': 0.0001, 'root': '/home/arthur/forcebalance/studies/001_water_tutorial', 'rigid_water': 0, 'search_tolerance': 0.0001, 'objective_history': 2, 'amoeba_polarization': 'direct', 'lm_guess': 1.0, 'priors': OrderedDict(), 'asynchronous': 0, 'read_pvals': None, 'backup': 1, 'jobtype': 'NEWTON', 'penalty_alpha': 0.001}

        for key in baseline_options.iterkeys():
            self.assertEqual(baseline_options[key],options[key])

        ## The force field component of the project
        forcefield  = FF(options)
        ## The objective function
        objective   = Objective(options, tgt_opts, forcefield)
        ## The optimizer component of the project
        optimizer   = Optimizer(options, objective, forcefield)
        ## Actually run the optimizer.
        optimizer.Run()

if __name__ == '__main__':
    unittest.main()