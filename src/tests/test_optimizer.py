from __future__ import absolute_import
import numpy
import forcebalance
import os, sys
import tarfile
import logging
import pytest

logger = logging.getLogger("test")

class TestOptimizer:
    @classmethod
    def setup_class(cls):
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, '../../studies/001_water_tutorial'))
        cls.input_file='very_simple.in'
        targets = tarfile.open('targets.tar.bz2','r')
        targets.extractall()
        targets.close()

        cls.options, cls.tgt_opts = forcebalance.parser.parse_inputs(cls.input_file)

        cls.options.update({'writechk':'checkfile.tmp'})

        cls.forcefield  = forcebalance.forcefield.FF(cls.options)
        cls.objective   = forcebalance.objective.Objective(cls.options, cls.tgt_opts, cls.forcefield)
        try: cls.optimizer   = forcebalance.optimizer.Optimizer(cls.options, cls.objective, cls.forcefield)
        except: pytest.fail("\nCouldn't create optimizer")

    @classmethod
    def teardown_class(cls):
        os.system('rm -rf result *.bak *.tmp')

    def test_optimizer(self):
        self.optimizer.writechk()
        assert os.path.isfile(self.options['writechk']), "Optimizer.writechk() didn't create expected file at %s " % options['writechk']
        read = self.optimizer.readchk()
        assert isinstance(read, dict)
